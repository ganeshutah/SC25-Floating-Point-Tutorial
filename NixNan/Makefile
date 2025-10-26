.PHONY: help setup priority analyze analyze-sass clean nixnan-baseline nixnan-priority

help:
	@echo "make setup          - Install"
	@echo "make priority       - Run key experiments"
	@echo "make analyze        - Analyze Python"
	@echo "make analyze-sass   - Analyze Python+SASS"
	@echo "make nixnan-baseline- Run baseline with SASS"
	@echo "make nixnan-priority- Run priority with SASS"
	@echo "make clean          - Clean"

setup:
	@./install.sh

priority:
	@python3 single_experiment.py --experiment bfloat16
	@python3 single_experiment.py --experiment attention_scale --scale 0.5
	@python3 single_experiment.py --experiment eager
	@python3 single_experiment.py --experiment attention_clip --clip 10.0
	@python3 analyze_results.py

analyze:
	@python3 analyze_results.py

analyze-sass:
	@python3 analyze_results.py --sass

nixnan-baseline:
	@./run_with_nixnan.sh baseline

nixnan-priority:
	@./run_with_nixnan.sh bfloat16
	@./run_with_nixnan.sh attention_scale --scale 0.5
	@./run_with_nixnan.sh eager
	@./run_with_nixnan.sh attention_clip --clip 10.0
	@python3 analyze_results.py --sass

clean:
	@rm -rf nan_experiments
