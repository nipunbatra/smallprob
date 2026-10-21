# Personal showcase

The static site is in `website/`. It pairs Smallprob model declarations with
equivalent, idiomatic Pyro declarations for seven small examples. Pyro's
broader inference and modeling capabilities are outside this comparison.
The diagrams and plots come from executed Python, and the notebooks identify
synthetic data, published observations, exact answers and VI approximations.

## Run locally

From the repository root:

```sh
uv sync --extra dev --extra vi --extra notebook --locked
.venv/bin/python -m examples.build_site
.venv/bin/python -m examples.check_site
python3 -m http.server 8820 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:8820`. There is no Node build, JavaScript package install,
remote font, analytics service or Python runtime in the browser. Code-copy
buttons copy Python for a local interpreter; the site does not execute it.

To update numerical notebook results, run the notebook commands from the README
before rebuilding the site. The site builder exports the saved executed
notebooks, redraws model diagrams and checks the probability-model comparisons;
it does not rerun the long VI fits. Commit generated `website/` files alongside
their sources so the same showcase works directly from a checkout.

## GitHub and publication

Repository: <https://github.com/nipunbatra/smallprob>.

The repository is initially private. A private repository does **not** make a
GitHub Pages site private. GitHub restricts access-controlled Pages to qualifying
Enterprise Cloud organization repositories; personal-account Pages are public.
See [GitHub's visibility documentation](https://docs.github.com/en/enterprise-cloud%40latest/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).

The workflow `.github/workflows/checks.yml` tests the library and builds the
showcase. Publication is deliberately disabled unless the repository variable
`ENABLE_PUBLIC_PAGES` is set to `true`. Enabling that variable is a decision to
publish the site publicly, including example code and downloadable notebooks.
Keep it unset for private development.

After public publication is approved, configure Pages to use GitHub Actions,
set the variable, and run the workflow. Private-source Pages may also require a
paid GitHub plan; never change the repository's visibility just to bypass this
requirement without the owner's approval. The expected project URL is
`https://nipunbatra.github.io/smallprob/`; it is only live after a successful
Pages deployment.
