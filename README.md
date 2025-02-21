# MAI-BIAS modules

[![Integration Tests](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/integration.yml/badge.svg)](https://github.com/mammoth-eu/mammoth-commons/actions/workflows/integration.yml)
![Coverage](./coverage-badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 

*Quickly develop and locally run MAI-BIAS toolkit modules.*

This repository holds the mammoth-commons library with supporting
datatypes and decorators shared by various fairness assessment modules.
It also hosts a catalogue of dataset loaders, model loaders, and fairness analysis 
and mitigation modules. It finally contains
a **desktop application** that runs the modules in your local machine.

![logo](demonstrator/logo.png)

## 🔬 Run locally

1. Download or clone this repository. Prefer working in a virtual environment.
2. Install dependencies with `pip install -r requirements[test].txt`. This can take a bit of time.
3. Launch the local app with `python demonstrator/app.py` or, if this fails on your platform, with `python -m demonstrator.app`.

<details><summary>VSCode launch profile</summary>  

```json 
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
        },
        {
            "name": "Python: Test",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Demonstrator",
            "type": "debugpy",
            "request": "launch",
            "module": "demonstrator.app",
            "justMyCode": false
        }
    ]
}
``` 
</details>
 
## 🖥 [Deploy in a server](https://github.com/mammoth-eu/mammoth-toolkit-releases)

## :clipboard: [Modules](https://mammoth-eu.github.io/mammoth-commons/)

## :thumbsup: [Contributing](CONTRIBUTING.md)
