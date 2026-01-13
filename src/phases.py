from enum import Enum

class TrainingPhase(Enum):
    UCB_ESTIMATION = "ucb_estimation" # Phase 1: UCB counts are updated, no physical pruning
    PRUNING_INFERENCE = "pruning_inference" # Phase 2: Physical pruning occurs, UCB counts are NOT updated