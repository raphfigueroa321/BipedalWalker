import pandas as pd
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_final_stats(experiment_dir):
    all_rewards = []
    all_lengths = []
    
    # Path to all seed folders inside the experiment
    seed_folders = glob.glob(f"{experiment_dir}/*/")
    
    for folder in seed_folders:
        # Find the tfevents file
        event_file = glob.glob(f"{folder}/*tfevents*")[0]
        acc = EventAccumulator(event_file)
        acc.Reload()
        
        # Extract the final scalar values
        reward = acc.Scalars('rollout/ep_rew_mean')[-1].value
        length = acc.Scalars('rollout/ep_len_mean')[-1].value
        
        all_rewards.append(reward)
        all_lengths.append(length)
    
    # Calculate Statistics
    df = pd.DataFrame({'Reward': all_rewards, 'Length': all_lengths})
    return {
        'Mean Reward': df['Reward'].mean(),
        'Std Reward': df['Reward'].std(),
        'Mean Length': df['Length'].mean()
    }

# Usage
stats = get_final_stats("logs/PPO_HullPenaltyV1/20251225-2325")
print(stats)