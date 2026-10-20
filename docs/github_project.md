# Managing FraudLens with a GitHub Project

A GitHub Project (the new "Projects", v2) is the team's single view of the work: every task is an **issue**
in the repository, the Project board shows where each one is, and pull requests move them to *Done*
automatically. The lifecycle stages are a field on the board, so progress can be tracked stage by stage.

```
project_plan.yaml ──► labels + milestones + issues (repository) ──► Project board (Status × Lifecycle stage)
                                                                     ▲
        branch ──► commits ──► pull request "Closes #12" ──► merged ─┘ (issue closed → Status = Done)
```

## 1. One-time setup (10 minutes)

### Option A: scripted (recommended)

1. Install the GitHub CLI (`winget install GitHub.cli`, or https://cli.github.com) and sign in with the
   Project scope:

   ```bash
   gh auth login --scopes "repo,project,workflow"
   ```

2. From the repository root, preview, then create everything from `scripts/github/project_plan.yaml`:

   ```bash
   python scripts/github/setup_project.py --owner Joys101 --dry-run
   python scripts/github/setup_project.py --owner Joys101
   ```

   `--owner` is the GitHub user (or organisation) that will own the board; use the repository owner
   (`anasua2905`) if they run it. The script creates the labels (`stage: …`, `type: …`, `priority: …`), the four
   milestones (M1-M4), one issue per task with an acceptance checklist, closes the finished ones, creates the
   Project **"FraudLens - PRT661 Assessment 4"**, adds a single-select field **Lifecycle stage**, adds every
   issue to the board with its Status and stage, and links the board to the repository. It is idempotent:
   edit the YAML and run it again to add new tasks.

### Option B: by hand

1. Repository → **Projects** → **New project** → *Board* template → name it "FraudLens - PRT661 Assessment 4".
2. **Settings** (⋯ → Settings) → **+ New field** → *Single select* "Lifecycle stage" with the options
   1 Problem understanding … 8 Monitoring & improvement, Data platform, Deliverables.
3. Repository → **Issues** → **Milestones**: create M1-M4 (see `project_plan.yaml`).
4. Create the issues (the *Lifecycle task* template asks for the stage, goal and acceptance criteria) and add
   them to the Project.

## 2. Views to create (in the Project, + New view)

| View | Layout | Settings | Use |
|---|---|---|---|
| **Board** | Board | Column by *Status* | daily stand-up: what is in progress, what is blocked |
| **By stage** | Table | Group by *Lifecycle stage*, sort by *Status* | lifecycle progress: how complete is each of the 8 stages |
| **Milestones** | Table | Group by *Milestone* | progress towards each deliverable (M1-M4) |
| **Roadmap** | Roadmap | Date fields *Start* / *Target* (add two Date fields), group by milestone | timeline for the supervisor |
| **My work** | Table | Filter `assignee:@me is:open` | each member's queue |

## 3. Built-in workflows (Project → ⋯ → Workflows)

Turn on:

* **Auto-add to project**: repository `fraudlens-or-credit-card-fraud-detection`, filter `is:issue,pr is:open`
  (new issues and PRs appear on the board automatically).
* **Item added to project** → Status *Todo*.
* **Item closed** → Status *Done*; **Pull request merged** → Status *Done*.
* **Item reopened** → Status *In Progress*.

## 4. Day-to-day: tracking step-by-step progress

1. **Pick a task**: on the Board, assign yourself and move it to *In Progress*.
2. **Branch** from `main` (or `final-a4` during the assessment), named after the issue: `12-clickhouse-benchmark`.
   (On the issue page, *Development → Create a branch* does this for you.)
3. **Commit** with messages that reference the issue (`Load gold into ClickHouse (#12)`). Tick the acceptance
   checklist in the issue as items are finished; the checklist progress shows on the board card.
4. **Pull request** with `Closes #12` in the description. The template asks for the lifecycle stage and the
   checks (tests, verification report, docs). CI (`ci.yml`, `platform.yml`) runs on every PR.
5. **Review and merge**: a teammate reviews; merging closes the issue and the workflow moves it to *Done*.
6. **Weekly**: open the *By stage* and *Milestones* views; milestone pages show the % complete. Use
   **Insights** (the chart icon in the Project) for a burn-up chart of closed items over time.

## 5. Conventions

| Label | Meaning |
|---|---|
| `stage: 1-problem` … `stage: 8-monitoring`, `stage: platform`, `stage: deliverable` | which part of the lifecycle the work belongs to |
| `type: feature / bug / docs / infra / analysis` | kind of work |
| `priority: high / medium` | ordering inside a column |

* One issue = one reviewable outcome (a stage output, a fix, a document); split anything bigger than ~2 days.
* Results-changing work must re-run the affected stages and keep `reports/verification_report.json` at PASS.
* Never commit data, `platform/.env` or credentials; artefacts > 100 MB do not belong in Git.

## 6. Current state

The plan file mirrors the actual project: stages 1-8 are complete, the CI workflows are installed and green
(including the Docker platform end to end and the OpenMetadata governance job), and `final-a4` was merged into
`main` through pull request #1. The open item is a permanently hosted dashboard. Re-run the setup script after
editing `project_plan.yaml` to keep the board in sync.
