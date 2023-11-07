#!/usr/bin/env sh
set -eu

# Gait Analyzer setup using `docker compose`.
# See https://github.com/abishekmuthian/gaitanalyzer for detailed installation steps.

check_dependencies() {
	if ! command -v curl > /dev/null; then
		echo "curl is not installed."
		exit 1
	fi

	if ! command -v docker > /dev/null; then
		echo "docker is not installed."
		exit 1
	fi

	if ! command -v docker compose > /dev/null; then
		echo "docker compose is not installed."
		exit 1
	fi

	if ! command -v nvidia-smi > /dev/null; then
		echo "nvidia drivers are not installed."
		exit 1
	fi

	if ! command -v docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi > /dev/null; then
		echo "nvidia container toolkit is not installed."
		exit 1
	fi    
}

setup_folders(){
	mkdir input_videos
	mkdir output_videos
}

setup_containers() {
	curl -o compose-gpu.yaml https://raw.githubusercontent.com/abishekmuthian/gaitanalyzer/main/compose-gpu.yaml
	docker compose -f compose-gpu.yaml up -d
}

run_model(){
    docker exec ollama ollama run llama2
}

show_output(){
	echo -e "\nGait Analyzer is now up and running. Visit http://localhost:8501 in your browser.\n"
}


check_dependencies
setup_folders
setup_containers
run_model
show_output