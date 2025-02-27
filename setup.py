from setuptools import setup, find_packages                                                                                                                                           
                                                                                                                                                                                       
setup(                                                                                                                                                                                
    name="llm_optimizer",                                                                                                                                                             
    packages=find_packages(),                                                                                                                                                         
    include_package_data=True,                                                                                                                                                        
    install_requires=[                                                                                                                                                                
        line.strip() for line in open("requirements.txt") if not line.startswith("#")                                                                                                 
    ],                                                                                                                                                                                
)     