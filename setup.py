from setuptools import setup, find_packages

setup(
    name="aigovivo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "Click",
    ],
    entry_points="""
        [console_scripts]
        aigovivo=aigovivo.cli:cli
    """,
)

# entry_points="""
#         [console_scripts]
#         h5=h5.h5_data:store_h5
#         corr=corr.corr_analysis:eras_corr
#         ml=ml.machine_learning:cli
#     """,
# 