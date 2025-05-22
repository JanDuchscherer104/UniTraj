ENV_NAME=unitraj
PYTHON_VERSION=3.11
CUDA_CHANNEL=nvidia/label/cuda-12.4.1

.PHONY: env-base env-cuda set-paths poetry install jupyter-kernel

help:
	@echo ""
	@echo "🚀 UniTraj Setup - Available Commands"
	@echo "--------------------------------------"
	@echo "make help              Show this help message"
	@echo "make install           Full install: env + CUDA + poetry deps"
	@echo "make install-dev       Full install: env + CUDA + poetry deps + dev deps"
	@echo "make env-base          Create Conda env and install Poetry"
	@echo "make env-cuda          Install and verify CUDA Toolkit"
	@echo "make set-paths         Append CUDA env vars to ~/.bashrc"
	@echo "make poetry            Install project dependencies via Poetry"
	@echo "make jupyter-kernel    Register Conda env as a Jupyter kernel"
	@echo ""

env-base:
	@echo "📦 Creating Conda environment '$(ENV_NAME)' with Python $(PYTHON_VERSION)..."
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)
	@echo "✅ Environment '$(ENV_NAME)' created."

	@echo "🎩 Installing Poetry..."
	conda run -n $(ENV_NAME) conda install -y -c conda-forge poetry
	@echo "✅ Poetry installed."

	@echo "🧠 Enabling Poetry autocompletion for Bash..."
	conda run -n $(ENV_NAME) poetry completions bash >> ~/.bash_completion
	grep -qxF '[[ -r ~/.bash_completion ]] && . ~/.bash_completion' ~/.bashrc || echo '[[ -r ~/.bash_completion ]] && . ~/.bash_completion' >> ~/.bashrc

env-cuda:
	@echo "🔧 Installing CUDA Toolkit from $(CUDA_CHANNEL)..."
	conda run -n $(ENV_NAME) conda install -y -c $(CUDA_CHANNEL) cuda-toolkit

	@echo "⚙️ Verifying CUDA installation..."
	conda run -n $(ENV_NAME) nvcc --version | tee /tmp/cuda_version.txt
	grep -q "release 12.4, V12.4.131" /tmp/cuda_version.txt || (echo "❌ CUDA version check failed. Aborting." && exit 1)
	@echo "✅ CUDA Toolkit correctly installed."

set-paths:
	@echo "🔧 Appending CUDA environment variables to ~/.bashrc..."
	echo 'export CUDA_HOME=$CONDA_PREFIX' >> ~/.bashrc
	echo 'export PATH=$CUDA_HOME/bin:$$PATH' >> ~/.bashrc
	echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$$LD_LIBRARY_PATH' >> ~/.bashrc
	@echo "✅ Environment variables added. Please run 'source ~/.bashrc' or restart the shell."

poetry:
	@echo "📜 Running poetry install in environment '$(ENV_NAME)'..."
	conda run -n $(ENV_NAME) poetry install
	@echo "✅ Project dependencies installed via Poetry."

poetry-dev:
	@echo "📜 Running poetry install in environment '$(ENV_NAME)'..."
	conda run -n $(ENV_NAME) poetry install --with dev
	@echo "✅ Project dependencies installed via Poetry."

jupyter-kernel:
	@echo "🎓 Registering Conda environment '$(ENV_NAME)' as a Jupyter kernel..."
	conda run -n $(ENV_NAME) python -m ipykernel install --user --name=$(ENV_NAME) --display-name="Python ($(ENV_NAME))"
	@echo "✅ Kernel 'Python ($(ENV_NAME))' is now available in Jupyter."

install: env-base env-cuda poetry
	@echo "🚀 All setup steps completed. Your development environment '$(ENV_NAME)' is ready!"
	@echo ""
	@echo "💡 Next steps:"
	@echo "- Activate the Conda environment:     conda activate $(ENV_NAME)"
	@echo "- Enter the Poetry shell:            poetry shell"
	@echo "- Register environment as a Jupyter kernel: make jupyter-kernel"
	@echo ""

install-dev: env-base env-cuda poetry
	@echo "🚀 All setup steps completed. Your development environment '$(ENV_NAME)' is ready!"
	@echo ""
	@echo "💡 Next steps:"
	@echo "- Activate the Conda environment:     conda activate $(ENV_NAME)"
	@echo "- Enter the Poetry shell:            poetry shell"
	@echo "- Register environment as a Jupyter kernel: make jupyter-kernel"
	@echo ""