#!/bin/bash
source /home/mila/a/aghajohm/repos/qadetective/env/bin/activate
python run_league.py main --league_name=league_advantage_alignment_mask_2024_04_23 --debug_mode=False --agent1="loqa_rb_ablation_1tieobkn" --agent2="advantage_alignment_mask_cooperative_empathetic_5" --trace_length=16