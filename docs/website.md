# Smallprob showcase

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

Public repository: <https://github.com/nipunbatra/smallprob>.

Public showcase: <https://nipunbatra.github.io/smallprob/>.

The [GitHub Actions workflow](../.github/workflows/checks.yml) runs on pushes to
`main`, pull requests and manual dispatch. It installs the locked environment,
tests the library and model comparisons, builds the showcase from the saved
executed notebooks, and validates the generated site.

For publication, configure the repository's Pages source as **GitHub Actions**
and set the repository Actions variable `ENABLE_PUBLIC_PAGES` to `true`. On
`main`, a successful check job uploads the built `website/` artifact; the
dependent publish job deploys it to Pages. Pull requests run the checks without
publishing. Leaving the variable unset or setting it to `false` skips future
publication; it does not remove an already published site.

The Pages deployment serves the static showcase, example code and downloadable
notebooks publicly. It does not run Python or inference on a server. Check the
workflow's `publish` job and the `github-pages` environment for deployment
status.
