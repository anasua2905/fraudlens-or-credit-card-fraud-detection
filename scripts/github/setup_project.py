"""Create the FraudLens GitHub Project (v2), labels, milestones and issues from project_plan.yaml.

Idempotent: anything that already exists (same name/title) is reused, so it is safe to re-run after
editing the plan. Closed/open state and the board's Status follow each issue's `status`.

Authentication (one of):
  * GitHub CLI:  gh auth login --scopes "repo,project"     (the script calls `gh auth token`)
  * a token in GH_TOKEN or GITHUB_TOKEN (classic PAT with `repo` + `project`, or a fine-grained
    token with Issues/Metadata read-write on the repo and Projects read-write on the owner)

    python scripts/github/setup_project.py --repo anasua2905/fraudlens-or-credit-card-fraud-detection \
        --owner Joys101 --title "FraudLens - PRT661 Assessment 4"
    python scripts/github/setup_project.py ... --dry-run      # print what would be created
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

API = "https://api.github.com"
PLAN = Path(__file__).with_name("project_plan.yaml")
STATUS_OPTION = {"done": "Done", "in_progress": "In Progress", "todo": "Todo"}


def token() -> str:
    for k in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sys.exit("No GitHub token: run `gh auth login --scopes repo,project` or set GH_TOKEN")


class GitHub:
    def __init__(self, tok: str, dry: bool):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})
        self.dry = dry

    def rest(self, method: str, path: str, **kw):
        if self.dry and method != "GET":
            print(f"  [dry-run] {method} {path} {kw.get('json', '')}")
            return {}
        r = self.s.request(method, API + path, timeout=30, **kw)
        if r.status_code >= 400:
            raise SystemExit(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        time.sleep(0.3)                      # stay well inside the secondary rate limits
        return r.json() if r.text else {}

    def paged(self, path: str, **params):
        out, page = [], 1
        while True:
            batch = self.rest("GET", path, params={**params, "per_page": 100, "page": page})
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def gql(self, query: str, **variables):
        if self.dry and query.lstrip().startswith("mutation"):
            print(f"  [dry-run] {query.split('(')[0].split()[-1]} {variables}")
            return {}
        r = self.s.post(API + "/graphql", json={"query": query, "variables": variables}, timeout=30)
        data = r.json()
        if r.status_code >= 400 or data.get("errors"):
            raise SystemExit(f"GraphQL error: {data.get('errors') or r.text[:400]}")
        return data["data"]


def ensure_labels(gh: GitHub, repo: str, plan: dict) -> None:
    have = {l["name"] for l in gh.paged(f"/repos/{repo}/labels")}
    wanted = [dict(l) for l in plan["labels"]]
    wanted += [{"name": f"stage: {s['key']}", "color": "c5def5", "description": s["name"]} for s in plan["stages"]]
    for l in wanted:
        if l["name"] not in have:
            gh.rest("POST", f"/repos/{repo}/labels", json=l)
            print(f"label + {l['name']}")


def ensure_milestones(gh: GitHub, repo: str, plan: dict) -> dict[int, int]:
    have = {m["title"]: m["number"] for m in gh.paged(f"/repos/{repo}/milestones", state="all")}
    numbers = {}
    for i, m in enumerate(plan["milestones"], start=1):
        if m["title"] not in have:
            have[m["title"]] = gh.rest("POST", f"/repos/{repo}/milestones", json=m).get("number", 0)
            print(f"milestone + {m['title']}")
        numbers[i] = have[m["title"]]
    return numbers


def ensure_issues(gh: GitHub, repo: str, plan: dict, milestones: dict[int, int]) -> list[dict]:
    have = {i["title"]: i for i in gh.paged(f"/repos/{repo}/issues", state="all") if "pull_request" not in i}
    out = []
    for it in plan["issues"]:
        labels = [f"stage: {it['stage']}", f"type: {it['type']}"] + ([f"priority: {it['priority']}"] if it.get("priority") else [])
        body = it["body"].rstrip() + "\n\n_Tracked on the FraudLens GitHub Project; created by `scripts/github/setup_project.py`._"
        issue = have.get(it["title"])
        if issue is None:
            issue = gh.rest("POST", f"/repos/{repo}/issues", json={"title": it["title"], "body": body,
                                                                   "labels": labels, "milestone": milestones[it["milestone"]]})
            print(f"issue + {it['title']}")
        want_state = "closed" if it["status"] == "done" else "open"
        if issue and issue.get("state") != want_state:
            gh.rest("PATCH", f"/repos/{repo}/issues/{issue['number']}",
                    json={"state": want_state, **({"state_reason": "completed"} if want_state == "closed" else {})})
        out.append({**it, "node_id": issue.get("node_id"), "number": issue.get("number")})
    return out


def ensure_project(gh: GitHub, owner: str, title: str, repo: str, plan: dict, issues: list[dict]) -> str:
    q = """query($login: String!) {
      user(login: $login) { id projectsV2(first: 50) { nodes { id title url } } }
    }"""
    try:
        node = gh.gql(q, login=owner)["user"]
    except SystemExit:
        node = gh.gql(q.replace("user(", "organization("), login=owner)["organization"]
    project = next((p for p in node["projectsV2"]["nodes"] if p["title"] == title), None)
    if project is None:
        if gh.dry:
            print(f"[dry-run] would create project '{title}' for {owner}")
            return "(dry run)"
        project = gh.gql("""mutation($owner: ID!, $title: String!) {
            createProjectV2(input: {ownerId: $owner, title: $title}) { projectV2 { id title url } } }""",
                         owner=node["id"], title=title)["createProjectV2"]["projectV2"]
        print(f"project + {project['url']}")
    pid = project["id"]

    owner_name, repo_name = repo.split("/")
    repo_id = gh.gql("query($o: String!, $n: String!) { repository(owner: $o, name: $n) { id } }",
                     o=owner_name, n=repo_name)["repository"]["id"]
    try:
        gh.gql("mutation($p: ID!, $r: ID!) { linkProjectV2ToRepository(input: {projectId: $p, repositoryId: $r}) { repository { id } } }",
               p=pid, r=repo_id)
    except SystemExit as exc:                  # already linked, or not allowed for this repo
        print(f"(repository link skipped: {str(exc)[:120]})")

    fields_q = """query($p: ID!) { node(id: $p) { ... on ProjectV2 { fields(first: 50) { nodes {
        ... on ProjectV2SingleSelectField { id name options { id name } } } } } } }"""
    fields = {f["name"]: f for f in gh.gql(fields_q, p=pid)["node"]["fields"]["nodes"] if f}
    if "Lifecycle stage" not in fields:
        opts = [{"name": s["name"], "color": s["color"], "description": s["key"]} for s in plan["stages"]]
        gh.gql("""mutation($p: ID!, $opts: [ProjectV2SingleSelectFieldOptionInput!]) {
            createProjectV2Field(input: {projectId: $p, dataType: SINGLE_SELECT, name: "Lifecycle stage",
                                         singleSelectOptions: $opts}) { projectV2Field { ... on ProjectV2SingleSelectField { id } } } }""",
               p=pid, opts=opts)
        print("field + Lifecycle stage")
        fields = {f["name"]: f for f in gh.gql(fields_q, p=pid)["node"]["fields"]["nodes"] if f}
    status, stage = fields["Status"], fields["Lifecycle stage"]
    stage_opt = {s["key"]: next(o["id"] for o in stage["options"] if o["name"] == s["name"]) for s in plan["stages"]}
    status_opt = {o["name"]: o["id"] for o in status["options"]}

    for it in issues:
        item = gh.gql("""mutation($p: ID!, $c: ID!) { addProjectV2ItemById(input: {projectId: $p, contentId: $c}) { item { id } } }""",
                      p=pid, c=it["node_id"])["addProjectV2ItemById"]["item"]["id"]
        for field, option in ((status["id"], status_opt.get(STATUS_OPTION[it["status"]])), (stage["id"], stage_opt[it["stage"]])):
            if option:
                gh.gql("""mutation($p: ID!, $i: ID!, $f: ID!, $o: String!) {
                    updateProjectV2ItemFieldValue(input: {projectId: $p, itemId: $i, fieldId: $f,
                                                          value: {singleSelectOptionId: $o}}) { projectV2Item { id } } }""",
                       p=pid, i=item, f=field, o=option)
        print(f"board: #{it['number']} {it['title']} -> {STATUS_OPTION[it['status']]}")
    return project["url"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="anasua2905/fraudlens-or-credit-card-fraud-detection")
    ap.add_argument("--owner", required=True, help="user or organisation that owns the Project")
    ap.add_argument("--title", default="FraudLens - PRT661 Assessment 4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    gh = GitHub(token(), a.dry_run)
    ensure_labels(gh, a.repo, plan)
    ms = ensure_milestones(gh, a.repo, plan)
    issues = ensure_issues(gh, a.repo, plan, ms)
    if a.dry_run:
        print(f"[dry-run] {len(issues)} issues would be added to the project")
        return 0
    url = ensure_project(gh, a.owner, a.title, a.repo, plan, issues)
    print(f"\nDone. Project board: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
