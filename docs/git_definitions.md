# Git

## 2. Some git Commands
| `git status` | Shows which files are staged, unstaged, or untracked. |
| `git log --oneline` | View a simplified history of recent commits. |
| `git diff` | Shows the specific line changes made since the last commit. |
| `git checkout [branch]`| Switches your working directory to a different branch. |
| `git merge [branch]` | Joins the history of another branch into your current one. |

## 3. The Commit Process
Committing is a two-step process in Git to ensure you only save the changes you intend to.

1. **Staging (`git add`)**: This prepares your changes. You can add specific files (`git add file.py`) or all changes (`git add .`).
2. **Committing (`git commit`)**: This creates a permanent snapshot of the staged changes in your local history.
3. **Push (`git push`)**: Pushes your commit to the repo.

## 4. Our Project Workflow
To prevent "non-fast-forward" errors encountered during setup:
1. **Pull often**: Run `git pull origin main` daily to stay in sync with the team.
2. **Atomic Commits**: Keep commits small and focused on one specific task.
3. **Rebase over Merge**: Use `git pull --rebase` to keep a linear, clean history.

## 5. Branching Conventions
TBD