from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [x.strip() for x in f.readlines() if x.strip() and not x.startswith("#") and "git+" not in x]

setup(
    name="deepVogue",
    version="1.0",
    description="StyleGAN3-t base for latent-cinema and data-driven generative art",
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=requirements,
    test_suite="tests",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "deepvogue-train=deepVogue.train:main",
            "deepvogue-generate=deepVogue.generate:generate_images",
            "deepvogue-project=deepVogue.projector:run_projection",
            "deepvogue-prepare=deepVogue.dataset_tool.prepare:cli",
            "deepvogue-walk=deepVogue.walk:main",
            "deepvogue-blend=deepVogue.blend:main",
            "deepvogue-factors=deepVogue.factors:cli",
            "deepvogue-cinema=deepVogue.cinema:cli",
        ],
    },
    zip_safe=False,
)
