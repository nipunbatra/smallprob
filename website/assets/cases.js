window.SMALLPROB_CASES = [
  {
    "id": "normal-mean",
    "title": "One unknown mean",
    "eyebrow": "Start here",
    "method": "Exact Gaussian inference",
    "description": "An unknown mean and one noisy observation. Two lines specify the probability relationships.",
    "notes": [
      "Normal(mean, sd) uses standard deviation in Python; the mathematical diagram writes variance.",
      "The observation y = 2 is an illustrative input. The posterior calculation is analytic."
    ],
    "sources": [
      {
        "label": "Pyro automatic Gaussian guides",
        "url": "https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal"
      }
    ],
    "smallprob": "from smallprob import model, Normal\n\n@model\ndef normal_observation():\n    mu = Normal(0, 3)\n    y = Normal(mu, 1)",
    "pyro": "import pyro\nimport pyro.distributions as dist\n\ndef normal_observation(y=None):\n    mu = pyro.sample(\"mu\", dist.Normal(0., 3.))\n    pyro.sample(\"y\", dist.Normal(mu, 1.), obs=y)",
    "result": "Posterior mean of mu: 1.800\n95% marginal interval: [-0.059, 3.659]",
    "diagram": "assets/diagrams/normal-mean.svg",
    "plot": null,
    "download": "downloads/examples/normal-mean.py",
    "pyro_download": "downloads/examples/normal-mean-pyro.py",
    "smallprob_method": "Exact Gaussian inference",
    "pyro_method": "Approximate VI · full-covariance Gaussian guide",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "simple = normal_observation()\ngenerated = simple.simulate(draws=5, seed=7)\ny = 2.0  # Illustrative observed value; generated is prior predictive.",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\ngenerated = Predictive(normal_observation, num_samples=5)()\ny = torch.tensor(2.0)  # Illustrative observation."
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = simple.infer(y=y)",
        "pyro": "from pyro.infer import SVI, Trace_ELBO\nfrom pyro.infer.autoguide import AutoMultivariateNormal\nfrom pyro.optim import ClippedAdam\n\npyro.clear_param_store()\npyro.set_rng_seed(7)\nguide = AutoMultivariateNormal(normal_observation)\nsteps = 3000\noptimizer = ClippedAdam({\"lr\": 0.03, \"lrd\": 0.1 ** (1 / steps)})\nsvi = SVI(normal_observation, guide, optimizer,\n          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))\nfor _ in range(steps):\n    loss = svi.step(y=y)"
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "print(posterior.mean(\"mu\"), posterior.sd(\"mu\"))",
        "pyro": "pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples=4000)(y=y)\nsamples = draws[\"mu\"].reshape(-1)\nprint(samples.mean().item(), samples.std().item())"
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "simple = normal_observation()\ngenerated = simple.simulate(draws=5, seed=7)\ny = 2.0  # Illustrative observed value; generated is prior predictive.\n\nposterior = simple.infer(y=y)\n\nprint(posterior.mean(\"mu\"), posterior.sd(\"mu\"))\nsimple.diagram(observed=[\"y\"]).save(\"normal-mean-model.svg\")"
  },
  {
    "id": "linear-regression",
    "title": "A line through observations",
    "eyebrow": "Automatic observations",
    "method": "Exact Gaussian inference",
    "description": "Give x as an array. The priors are shared; the likelihood creates one observation per element.",
    "notes": [
      "Thirty synthetic observations, generated with intercept 1.5, slope −0.8 and noise SD 0.35.",
      "The displayed fit uses NumPy. Torch inputs select Torch execution; exact Gaussian inference also supports available Torch devices."
    ],
    "sources": [
      {
        "label": "Pyro automatic Gaussian guides",
        "url": "https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal"
      }
    ],
    "smallprob": "from smallprob import model, Normal\n\n@model\ndef regression(x):\n    intercept = Normal(0, 5)\n    slope = Normal(0, 5)\n    y = Normal(intercept + slope * x, 0.35)",
    "pyro": "import pyro\nimport pyro.distributions as dist\n\ndef regression(x, y=None):\n    intercept = pyro.sample(\"intercept\", dist.Normal(0., 5.))\n    slope = pyro.sample(\"slope\", dist.Normal(0., 5.))\n    with pyro.plate(\"observations\", len(x)):\n        pyro.sample(\"y\", dist.Normal(intercept + slope * x, 0.35), obs=y)",
    "result": "NumPy slope: -0.8583",
    "diagram": "assets/diagrams/linear-regression.svg",
    "plot": "assets/plots/linear-regression.svg",
    "download": "downloads/examples/linear-regression.py",
    "pyro_download": "downloads/examples/linear-regression-pyro.py",
    "smallprob_method": "Exact Gaussian inference",
    "pyro_method": "Approximate VI · full-covariance Gaussian guide",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "import numpy as np\n\nx = np.linspace(-2, 2, 30)\nline = regression(x)\ndata = line.simulate(seed=7, fixed={\"intercept\": 1.5, \"slope\": -0.8})\ny = data[\"y\"][0]",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\nx = torch.linspace(-2, 2, 30)\ntruth = {\"intercept\": torch.tensor(1.5), \"slope\": torch.tensor(-0.8)}\ngenerator = pyro.poutine.condition(regression, data=truth)\ndata = Predictive(generator, num_samples=1)(x)\ny = data[\"y\"][0]"
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = line.infer(y=y)",
        "pyro": "from pyro.infer import SVI, Trace_ELBO\nfrom pyro.infer.autoguide import AutoMultivariateNormal\nfrom pyro.optim import ClippedAdam\n\npyro.clear_param_store()\npyro.set_rng_seed(7)\nguide = AutoMultivariateNormal(regression)\nsteps = 3000\noptimizer = ClippedAdam({\"lr\": 0.03, \"lrd\": 0.1 ** (1 / steps)})\nsvi = SVI(regression, guide, optimizer,\n          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))\nfor _ in range(steps):\n    loss = svi.step(x, y=y)"
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "print(posterior.mean(\"slope\"), posterior.interval(\"slope\"))\ndraws = posterior.sample(4000, seed=12)\nx_new = np.linspace(-3, 3, 50)\nmean_response = draws[\"intercept\"][:, None] + draws[\"slope\"][:, None] * x_new\nprint(np.quantile(mean_response, [0.05, 0.95], axis=0))\n# A band for the mean response; add observation noise for predictive draws.",
        "pyro": "pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples=4000)(x, y=y)\nslopes = draws[\"slope\"].reshape(-1)\nintercepts = draws[\"intercept\"].reshape(-1)\nprint(slopes.mean().item(), torch.quantile(slopes, torch.tensor([0.025, 0.975])))\nx_new = torch.linspace(-3, 3, 50)\nmean_response = intercepts[:, None] + slopes[:, None] * x_new\nprint(torch.quantile(mean_response, torch.tensor([0.05, 0.95]), dim=0))\n# A band for the mean response; add observation noise for predictive draws."
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "import numpy as np\n\nx = np.linspace(-2, 2, 30)\nline = regression(x)\ndata = line.simulate(seed=7, fixed={\"intercept\": 1.5, \"slope\": -0.8})\ny = data[\"y\"][0]\n\nposterior = line.infer(y=y)\n\nprint(posterior.mean(\"slope\"), posterior.interval(\"slope\"))\ndraws = posterior.sample(4000, seed=12)\nx_new = np.linspace(-3, 3, 50)\nmean_response = draws[\"intercept\"][:, None] + draws[\"slope\"][:, None] * x_new\nprint(np.quantile(mean_response, [0.05, 0.95], axis=0))\n# A band for the mean response; add observation noise for predictive draws.\nline.diagram(labels={\"intercept\": \"b\", \"slope\": \"m\"}, observed=[\"y\"]).save(\"linear-regression-model.svg\")"
  },
  {
    "id": "robust-regression",
    "title": "A line with heavier tails",
    "eyebrow": "Unknown noise",
    "method": "Approximate variational inference",
    "description": "Replace Normal with StudentT and give the positive scale a prior. The observation syntax stays the same.",
    "notes": [
      "Synthetic Normal observations with two deliberately added outliers. This one dataset illustrates the model, not a general robustness guarantee.",
      "StudentT scale is not its SD. With df = 4, variance is twice scale squared.",
      "VI intervals describe the fitted guide; they are not exact posterior intervals. This VI implementation uses Pyro and PyTorch internally.",
      "The saved result uses guide=\"full\" (Pyro-backed VI). Choose guide=\"diagonal\" for Smallprob's independent Normal guide implemented directly in Torch; it cannot learn posterior correlations."
    ],
    "sources": [
      {
        "label": "Pyro automatic Gaussian guides",
        "url": "https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal"
      }
    ],
    "smallprob": "from smallprob import model, Normal, HalfNormal, StudentT\n\n@model\ndef robust_regression(x):\n    b = Normal(0, 5)\n    m = Normal(0, 5)\n    sigma = HalfNormal(1)\n    y = StudentT(df=4, loc=b + m * x, scale=sigma)",
    "pyro": "import pyro\nimport pyro.distributions as dist\n\ndef robust_regression(x, y=None):\n    b = pyro.sample(\"b\", dist.Normal(0., 5.))\n    m = pyro.sample(\"m\", dist.Normal(0., 5.))\n    sigma = pyro.sample(\"sigma\", dist.HalfNormal(1.))\n    with pyro.plate(\"observations\", len(x)):\n        pyro.sample(\"y\", dist.StudentT(4., b + m * x, sigma), obs=y)",
    "result": "Approximate VI slope: -0.8772\n95% guide interval: (-0.9896, -0.7647)\nLearned Student t scale: 0.3346",
    "diagram": "assets/diagrams/robust-regression.svg",
    "plot": "assets/plots/robust-regression.svg",
    "download": "downloads/examples/robust-regression.py",
    "pyro_download": "downloads/examples/robust-regression-pyro.py",
    "smallprob_method": "Approximate VI · full-covariance Gaussian guide",
    "pyro_method": "Approximate VI · full-covariance Gaussian guide",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "import numpy as np\n\nx = np.linspace(-2, 2, 30)\nrobust = robust_regression(x)\ngenerated = robust.simulate(seed=7)\n# Fit the same outlier-contaminated fixture in both implementations.\ny = 1.5 - 0.8 * x + np.random.default_rng(7).normal(0, 0.35, len(x))\ny[[5, 24]] += [3.0, -3.0]",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\nimport numpy as np\n\nx_np = np.linspace(-2, 2, 30)\nx = torch.as_tensor(x_np)\ngenerated = Predictive(robust_regression, num_samples=1)(x)\ny_np = 1.5 - 0.8 * x_np + np.random.default_rng(7).normal(0, 0.35, len(x_np))\ny_np[[5, 24]] += [3.0, -3.0]\ny = torch.as_tensor(y_np)"
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = robust.infer(y=y, method=\"vi\", guide=\"full\", steps=3000, seed=7)",
        "pyro": "from pyro.infer import SVI, Trace_ELBO\nfrom pyro.infer.autoguide import AutoMultivariateNormal\nfrom pyro.optim import ClippedAdam\n\npyro.clear_param_store()\npyro.set_rng_seed(7)\nguide = AutoMultivariateNormal(robust_regression)\nsteps = 3000\noptimizer = ClippedAdam({\"lr\": 0.03, \"lrd\": 0.1 ** (1 / steps)})\nsvi = SVI(robust_regression, guide, optimizer,\n          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))\nfor _ in range(steps):\n    loss = svi.step(x, y=y)"
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "print(posterior.mean(\"m\"), posterior.interval(\"m\"))",
        "pyro": "pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples=4000)(x, y=y)\nsamples = draws[\"m\"].reshape(-1)\nprint(samples.mean().item(), samples.std().item())\nprint(torch.quantile(samples, torch.tensor([0.025, 0.975])))"
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "import numpy as np\n\nx = np.linspace(-2, 2, 30)\nrobust = robust_regression(x)\ngenerated = robust.simulate(seed=7)\n# Fit the same outlier-contaminated fixture in both implementations.\ny = 1.5 - 0.8 * x + np.random.default_rng(7).normal(0, 0.35, len(x))\ny[[5, 24]] += [3.0, -3.0]\n\nposterior = robust.infer(y=y, method=\"vi\", guide=\"full\", steps=3000, seed=7)\n\nprint(posterior.mean(\"m\"), posterior.interval(\"m\"))\nrobust.diagram(observed=[\"y\"]).save(\"robust-regression-model.svg\")"
  },
  {
    "id": "hidden-markov-model",
    "title": "States, transitions, emissions",
    "eyebrow": "No time loop",
    "method": "Exact hidden-state inference",
    "description": "Supply the initial probabilities, transition table and Normal emissions. HMM constructs the time dependence.",
    "notes": [
      "Thirty synthetic time steps. State names label integer states 0 and 1; transition rows are current states and columns are next states.",
      "Transitions and emissions are fixed. Exact forward–backward inference recovers state probabilities on NumPy/CPU; learning parameters is not supported.",
      "The Pyro pane expresses this same joint law with named time sites. Pyro also has specialized HMM distributions and enumeration tools; this is not a port of its richer piano-sequence tutorial."
    ],
    "sources": [
      {
        "label": "Pyro HMM tutorial (broader scope)",
        "url": "https://pyro.ai/examples/hmm.html"
      },
      {
        "label": "Pyro exact state marginals",
        "url": "https://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.traceenum_elbo.TraceEnum_ELBO.compute_marginals"
      }
    ],
    "smallprob": "from smallprob import HMM, Normal\n\nchain = HMM(\n    states=[\"quiet\", \"active\"],\n    initial=[0.5, 0.5],\n    transition=[[0.9, 0.1], [0.15, 0.85]],\n    emissions=Normal([-1.5, 1.5], 0.45),\n    steps=30,\n)",
    "pyro": "import torch\nimport pyro\nimport pyro.distributions as dist\n\ndef hidden_markov_model(y=None, steps=30):\n    initial = torch.tensor([0.5, 0.5])\n    transition = torch.tensor([[0.9, 0.1], [0.15, 0.85]])\n    means = torch.tensor([-1.5, 1.5])\n    for t in pyro.markov(range(steps)):\n        probs = initial if t == 0 else transition[z]\n        z = pyro.sample(f\"z[{t}]\", dist.Categorical(probs))\n        observed = None if y is None else y[t]\n        pyro.sample(f\"y[{t}]\", dist.Normal(means[z], 0.45), obs=observed)",
    "result": "30 observations → 30 × 2 exact posterior state probabilities. The plot compares them with the generated states.",
    "diagram": "assets/diagrams/hidden-markov-model.svg",
    "plot": "assets/plots/hidden-markov-model.svg",
    "download": "downloads/examples/hidden-markov-model.py",
    "pyro_download": "downloads/examples/hidden-markov-model-pyro.py",
    "smallprob_method": "Exact hidden-state inference",
    "pyro_method": "Exact enumeration of hidden states",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "data = chain.simulate(seed=7)\ny = data[\"y\"][0]",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\ndata = Predictive(hidden_markov_model, num_samples=1)(steps=30)\ny = torch.stack([data[f\"y[{t}]\"][0] for t in range(30)])"
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = chain.infer(y=y)",
        "pyro": "from pyro.infer import TraceEnum_ELBO, config_enumerate\n\nenumerated = config_enumerate(hidden_markov_model, \"parallel\")\nobjective = TraceEnum_ELBO(max_plate_nesting=0)\nmarginals = objective.compute_marginals(enumerated, lambda **kwargs: None, y=y)\n# There are no unknown continuous parameters, so no optimizer is needed."
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "probabilities = posterior.probabilities(\"z\")\nprint(probabilities[:, 1])",
        "pyro": "probabilities = torch.stack([marginals[f\"z[{t}]\"].probs.reshape(2) for t in range(30)])\nprint(probabilities[:, 1])"
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "data = chain.simulate(seed=7)\ny = data[\"y\"][0]\n\nposterior = chain.infer(y=y)\n\nprobabilities = posterior.probabilities(\"z\")\nprint(probabilities[:, 1])\nchain.diagram(observed=[\"y\"]).save(\"hidden-markov-model.svg\")"
  },
  {
    "id": "weighing-scale",
    "title": "How much does it weigh?",
    "eyebrow": "From the Pyro tutorials",
    "method": "Exact Gaussian inference",
    "description": "A prior guess of 8.5 and a noisy reading of 9.5. Condition the same model in one call.",
    "notes": [
      "Same probability model as the original scale example: prior SD 1 and measurement SD 0.75.",
      "Pyro labels this older tutorial deprecated. The analytic posterior is Normal with mean 9.14 and SD 0.60."
    ],
    "sources": [
      {
        "label": "Pyro's scale example",
        "url": "https://pyro.ai/examples/intro_part_ii.html"
      },
      {
        "label": "Pyro automatic Gaussian guides",
        "url": "https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal"
      }
    ],
    "smallprob": "from smallprob import model, Normal\n\n@model\ndef weighing_scale(guess=8.5):\n    weight = Normal(guess, 1)\n    measurement = Normal(weight, 0.75)",
    "pyro": "import pyro\nimport pyro.distributions as dist\n\ndef weighing_scale(guess=8.5, measurement=None):\n    weight = pyro.sample(\"weight\", dist.Normal(guess, 1.))\n    pyro.sample(\"measurement\", dist.Normal(weight, 0.75), obs=measurement)",
    "result": "Exact posterior: mean = 9.14, SD = 0.60",
    "diagram": "assets/diagrams/weighing-scale.svg",
    "plot": "assets/plots/weighing-scale.svg",
    "download": "downloads/examples/weighing-scale.py",
    "pyro_download": "downloads/examples/weighing-scale-pyro.py",
    "smallprob_method": "Exact Gaussian inference",
    "pyro_method": "Approximate VI · full-covariance Gaussian guide",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "scale = weighing_scale()\ngenerated = scale.simulate(draws=5, seed=7)\nmeasurement = 9.5  # Illustrative reading; generated is prior predictive.",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\ngenerated = Predictive(weighing_scale, num_samples=5)()\nmeasurement = torch.tensor(9.5)  # Illustrative reading."
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = scale.infer(measurement=measurement)",
        "pyro": "from pyro.infer import SVI, Trace_ELBO\nfrom pyro.infer.autoguide import AutoMultivariateNormal\nfrom pyro.optim import ClippedAdam\n\npyro.clear_param_store()\npyro.set_rng_seed(7)\nguide = AutoMultivariateNormal(weighing_scale)\nsteps = 3000\noptimizer = ClippedAdam({\"lr\": 0.03, \"lrd\": 0.1 ** (1 / steps)})\nsvi = SVI(weighing_scale, guide, optimizer,\n          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))\nfor _ in range(steps):\n    loss = svi.step(measurement=measurement)"
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "print(posterior.mean(\"weight\"), posterior.sd(\"weight\"))",
        "pyro": "pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples=4000)(measurement=measurement)\nsamples = draws[\"weight\"].reshape(-1)\nprint(samples.mean().item(), samples.std().item())"
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "scale = weighing_scale()\ngenerated = scale.simulate(draws=5, seed=7)\nmeasurement = 9.5  # Illustrative reading; generated is prior predictive.\n\nposterior = scale.infer(measurement=measurement)\n\nprint(posterior.mean(\"weight\"), posterior.sd(\"weight\"))\nscale.diagram(labels={\"weight\": \"w\", \"measurement\": \"y\"}, observed=[\"measurement\"]).save(\"weighing-scale-model.svg\")"
  },
  {
    "id": "interaction-regression",
    "title": "Two groups, two slopes",
    "eyebrow": "Adapted from Pyro",
    "method": "Approximate variational inference",
    "description": "An intercept, two predictors and their interaction. Group 1 has slope + interaction; group 0 has slope.",
    "notes": [
      "Adaptation: fixed synthetic data replace GDP data, and HalfNormal(1) replaces Uniform(0, 10) for noise SD. The prior change changes the model.",
      "Both code panes use the adapted prior. The plot shows 90% guide bands for mean responses, excluding new observation noise.",
      "Against independent numerical integration, the saved fit's largest mean error is 0.126 reference SD and largest SD error is 9.5%. These checks do not establish accurate tails.",
      "The saved result uses guide=\"full\" (Pyro-backed VI). Choose guide=\"diagonal\" for Smallprob's independent Normal guide implemented directly in Torch; it cannot learn posterior correlations."
    ],
    "sources": [
      {
        "label": "Pyro's regression tutorial",
        "url": "https://pyro.ai/examples/intro_long.html#example-from-maximum-likelihood-regression-to-bayesian-regression"
      },
      {
        "label": "Pyro automatic Gaussian guides",
        "url": "https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal"
      }
    ],
    "smallprob": "from smallprob import model, Normal, HalfNormal\n\n@model\ndef interaction_regression(group, x):\n    intercept = Normal(0, 10)\n    group_effect = Normal(0, 1)\n    slope = Normal(0, 1)\n    interaction = Normal(0, 1)\n    sigma = HalfNormal(1)\n    y = Normal(intercept + group_effect * group + slope * x + interaction * group * x, sigma)",
    "pyro": "import pyro\nimport pyro.distributions as dist\n\ndef interaction_regression(group, x, y=None):\n    intercept = pyro.sample(\"intercept\", dist.Normal(0., 10.))\n    group_effect = pyro.sample(\"group_effect\", dist.Normal(0., 1.))\n    slope = pyro.sample(\"slope\", dist.Normal(0., 1.))\n    interaction = pyro.sample(\"interaction\", dist.Normal(0., 1.))\n    sigma = pyro.sample(\"sigma\", dist.HalfNormal(1.))\n    mean = intercept + group_effect * group + slope * x + interaction * group * x\n    with pyro.plate(\"observations\", len(x)):\n        pyro.sample(\"y\", dist.Normal(mean, sigma), obs=y)",
    "result": "Interaction mean: 0.833\n95% guide interval: [0.487, 1.180]\nLearned noise SD: 0.526",
    "diagram": "assets/diagrams/interaction-regression.svg",
    "plot": "assets/plots/interaction-regression.svg",
    "download": "downloads/examples/interaction-regression.py",
    "pyro_download": "downloads/examples/interaction-regression-pyro.py",
    "smallprob_method": "Approximate VI · full-covariance Gaussian guide",
    "pyro_method": "Approximate VI · full-covariance Gaussian guide",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "import numpy as np\n\nx = np.tile(np.linspace(-1.5, 1.5, 20), 2)\ngroup = np.repeat([0., 1.], 20)\nregression = interaction_regression(group, x)\ntruth = {\"intercept\": 1.2, \"group_effect\": 0.7,\n         \"slope\": -0.6, \"interaction\": 0.9, \"sigma\": 0.5}\ndata = regression.simulate(seed=103, fixed=truth)\ny = data[\"y\"][0]",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\nx = torch.linspace(-1.5, 1.5, 20).repeat(2)\ngroup = torch.tensor([0., 1.]).repeat_interleave(20)\ntruth = {name: torch.tensor(value) for name, value in {\n    \"intercept\": 1.2, \"group_effect\": 0.7,\n    \"slope\": -0.6, \"interaction\": 0.9, \"sigma\": 0.5}.items()}\ngenerator = pyro.poutine.condition(interaction_regression, data=truth)\npyro.set_rng_seed(103)\ndata = Predictive(generator, num_samples=1)(group, x)\ny = data[\"y\"][0]"
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = regression.infer(y=y, method=\"vi\", guide=\"full\", steps=4000, seed=11)",
        "pyro": "from pyro.infer import SVI, Trace_ELBO\nfrom pyro.infer.autoguide import AutoMultivariateNormal\nfrom pyro.optim import ClippedAdam\n\npyro.clear_param_store()\npyro.set_rng_seed(11)\nguide = AutoMultivariateNormal(interaction_regression)\nsteps = 4000\noptimizer = ClippedAdam({\"lr\": 0.03, \"lrd\": 0.1 ** (1 / steps)})\nsvi = SVI(interaction_regression, guide, optimizer,\n          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))\nfor _ in range(steps):\n    loss = svi.step(group, x, y=y)"
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "draws = posterior.sample(4000, seed=12)\ngroup_0_slope = draws[\"slope\"]\ngroup_1_slope = draws[\"slope\"] + draws[\"interaction\"]\nprint(group_0_slope.mean(), group_1_slope.mean())\nprint(posterior.mean(\"interaction\"), posterior.interval(\"interaction\"))",
        "pyro": "pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples=4000)(group, x, y=y)\ngroup_0_slope = draws[\"slope\"].reshape(-1)\ngroup_1_slope = group_0_slope + draws[\"interaction\"].reshape(-1)\nprint(group_0_slope.mean().item(), group_1_slope.mean().item())\nsamples = draws[\"interaction\"].reshape(-1)\nprint(samples.mean().item(), torch.quantile(samples, torch.tensor([0.025, 0.975])))"
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "import numpy as np\n\nx = np.tile(np.linspace(-1.5, 1.5, 20), 2)\ngroup = np.repeat([0., 1.], 20)\nregression = interaction_regression(group, x)\ntruth = {\"intercept\": 1.2, \"group_effect\": 0.7,\n         \"slope\": -0.6, \"interaction\": 0.9, \"sigma\": 0.5}\ndata = regression.simulate(seed=103, fixed=truth)\ny = data[\"y\"][0]\n\nposterior = regression.infer(y=y, method=\"vi\", guide=\"full\", steps=4000, seed=11)\n\ndraws = posterior.sample(4000, seed=12)\ngroup_0_slope = draws[\"slope\"]\ngroup_1_slope = draws[\"slope\"] + draws[\"interaction\"]\nprint(group_0_slope.mean(), group_1_slope.mean())\nprint(posterior.mean(\"interaction\"), posterior.interval(\"interaction\"))\nregression.diagram(labels={\"intercept\": \"b\", \"slope\": \"m\", \"group_effect\": \"a\", \"interaction\": \"c\"}, observed=[\"y\"]).save(\"interaction-regression-model.svg\")"
  },
  {
    "id": "eight-schools",
    "title": "Eight schools, shared information",
    "eyebrow": "Adapted from Pyro",
    "method": "Approximate variational inference",
    "description": "Eight effects share an unknown population mean and scale. An array of local effects handles the hierarchy.",
    "notes": [
      "The eight observations and standard errors are the published Eight Schools data, not synthetic data.",
      "Adaptation: HalfNormal(5) replaces Pyro's HalfCauchy(25) prior on tau, and VI replaces NUTS. Both code panes use the adapted law; its posterior differs from the original example.",
      "School effects come from joint draws of mu, tau and eta. Against numerical integration, hyperparameter mean errors are at most 0.083 reference SD and SD errors at most 7.4%; intervals remain approximate.",
      "The saved result uses guide=\"full\" (Pyro-backed VI). Choose guide=\"diagonal\" for Smallprob's independent Normal guide implemented directly in Torch; it cannot learn posterior correlations."
    ],
    "sources": [
      {
        "label": "Pyro's Eight Schools model",
        "url": "https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/mcmc.py"
      },
      {
        "label": "Published observations",
        "url": "https://github.com/pyro-ppl/pyro/blob/1.9.1/examples/eight_schools/data.py"
      },
      {
        "label": "Pyro automatic Gaussian guides",
        "url": "https://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoMultivariateNormal"
      }
    ],
    "smallprob": "import numpy as np\nfrom smallprob import model, Normal, HalfNormal\n\n@model\ndef eight_schools(se):\n    mu = Normal(0, 10)\n    tau = HalfNormal(5)\n    eta = Normal(np.zeros(len(se)), 1)\n    y = Normal(mu + tau * eta, se)",
    "pyro": "import torch\nimport pyro\nimport pyro.distributions as dist\n\ndef eight_schools(se, y=None):\n    mu = pyro.sample(\"mu\", dist.Normal(0., 10.))\n    tau = pyro.sample(\"tau\", dist.HalfNormal(5.))\n    with pyro.plate(\"schools\", len(se)):\n        eta = pyro.sample(\"eta\", dist.Normal(torch.zeros_like(se), 1.))\n        pyro.sample(\"y\", dist.Normal(mu + tau * eta, se), obs=y)",
    "result": "Population mean: 6.498\nBetween-school scale: 3.072",
    "diagram": "assets/diagrams/eight-schools.svg",
    "plot": "assets/plots/eight-schools.svg",
    "download": "downloads/examples/eight-schools.py",
    "pyro_download": "downloads/examples/eight-schools-pyro.py",
    "smallprob_method": "Approximate VI · full-covariance Gaussian guide",
    "pyro_method": "Approximate VI · full-covariance Gaussian guide",
    "workflows": [
      {
        "id": "generate",
        "label": "Generate data",
        "smallprob": "se = np.array([15., 10., 16., 11., 9., 11., 10., 18.])\ny = np.array([28., 8., -3., 7., -1., 1., 18., 12.])\nschools = eight_schools(se)\ngenerated = schools.simulate(draws=2, seed=7)\n# Fit the published observations, not the prior-predictive draw.",
        "pyro": "import torch\nfrom pyro.infer import Predictive\n\ntorch.set_default_dtype(torch.float64)\npyro.set_rng_seed(7)\n\nse = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])\ny = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])\ngenerated = Predictive(eight_schools, num_samples=2)(se)\n# Fit the published observations, not the prior-predictive draw."
      },
      {
        "id": "fit",
        "label": "Fit the model",
        "smallprob": "posterior = schools.infer(y=y, method=\"vi\", guide=\"full\", steps=4000, seed=11)",
        "pyro": "from pyro.infer import SVI, Trace_ELBO\nfrom pyro.infer.autoguide import AutoMultivariateNormal\nfrom pyro.optim import ClippedAdam\n\npyro.clear_param_store()\npyro.set_rng_seed(11)\nguide = AutoMultivariateNormal(eight_schools)\nsteps = 4000\noptimizer = ClippedAdam({\"lr\": 0.03, \"lrd\": 0.1 ** (1 / steps)})\nsvi = SVI(eight_schools, guide, optimizer,\n          loss=Trace_ELBO(num_particles=32, vectorize_particles=True))\nfor _ in range(steps):\n    loss = svi.step(se, y=y)"
      },
      {
        "id": "summarize",
        "label": "Use the posterior",
        "smallprob": "draws = posterior.sample(6000, seed=12)\neffects = draws[\"mu\"][:, None] + draws[\"tau\"][:, None] * draws[\"eta\"]\nprint(effects.mean(axis=0))",
        "pyro": "pyro.set_rng_seed(12)\ndraws = Predictive(guide, num_samples=6000)(se, y=y)\neffects = draws[\"mu\"].reshape(-1, 1) + draws[\"tau\"].reshape(-1, 1) * draws[\"eta\"]\nprint(effects.mean(dim=0))"
      }
    ],
    "workflow_note": "Run each column from Model through Use the posterior, or download its complete script. The generators use different random-number streams: equal seeds do not imply equal data. The displayed plots and saved summaries come from the Smallprob notebook; inference checks use shared observations. Pyro VI summaries below use Monte Carlo guide draws; Smallprob's marginal moments and intervals are analytic properties of its fitted guide.",
    "run": "se = np.array([15., 10., 16., 11., 9., 11., 10., 18.])\ny = np.array([28., 8., -3., 7., -1., 1., 18., 12.])\nschools = eight_schools(se)\ngenerated = schools.simulate(draws=2, seed=7)\n# Fit the published observations, not the prior-predictive draw.\n\nposterior = schools.infer(y=y, method=\"vi\", guide=\"full\", steps=4000, seed=11)\n\ndraws = posterior.sample(6000, seed=12)\neffects = draws[\"mu\"][:, None] + draws[\"tau\"][:, None] * draws[\"eta\"]\nprint(effects.mean(axis=0))\nschools.diagram(observed=[\"y\"]).save(\"eight-schools-model.svg\")"
  }
];
