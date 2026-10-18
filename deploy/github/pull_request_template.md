## What and why

<!-- one or two sentences; link the issue: "Closes #12" moves it to Done on the Project board -->

## Lifecycle stage(s)

- [ ] 1 Problem  - [ ] 2 Acquisition  - [ ] 3 Preparation  - [ ] 4 EDA
- [ ] 5 Modelling  - [ ] 6 Evaluation  - [ ] 7 Deployment  - [ ] 8 Monitoring  - [ ] Platform / docs

## Checks

- [ ] `pytest` passes locally (or CI is green)
- [ ] If results changed: pipeline re-run, `reports/verification_report.json` is PASS, bundle rebuilt
- [ ] Notebooks / docs updated where behaviour changed
- [ ] No secrets, data files or `platform/.env` committed
