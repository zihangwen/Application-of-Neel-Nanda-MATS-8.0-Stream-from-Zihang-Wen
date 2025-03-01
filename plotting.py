# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %%
with open("data/answer_dict_truncate_paragraph.pkl", "rb") as f:
    answer_dict_reload = pickle.load(f)

# %%
def paragraph_num_keeps(answer_dict):
    for key, value in answer_dict.items():
        num_of_removal = len(value)
        for i in range(num_of_removal):
            value[i][2] = i + 1
    return answer_dict

# %%
answer_dict = paragraph_num_keeps(answer_dict_reload)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Sample data with added hypothetical example
data = answer_dict_reload

# Function to normalize answer values for comparison
def normalize_answer(answer):
    if answer is None:
        return None
        
    try:
        # Try to convert to float first (handles decimal strings)
        float_value = float(answer)
        
        # Check if it's an integer value
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except (ValueError, TypeError):
        # If conversion fails, return the original answer
        return answer

# Function to find examples where original was wrong but paragraph removal led to correct answers
def find_wrong_to_correct_examples(data):
    examples = {}
    
    for question_id, question_data in data.items():
        true_answer = normalize_answer(question_data[0][0])  # The true answer
        original_answer = normalize_answer(question_data[0][1])  # The original tested answer
        
        # Check if the original answer was wrong
        if true_answer != original_answer:
            # Check each trial to see if paragraph removal led to correct answers
            correct_trials = []
            for trial in question_data:
                paragraph_keeps = trial[2]  # Number of paragraph keeps
                removed_answer = normalize_answer(trial[3])  # Answer after removal
                
                if removed_answer == true_answer:
                    correct_trials.append({
                        'paragraph_keeps': paragraph_keeps,
                        'answer': removed_answer
                    })
            
            if correct_trials:
                examples[question_id] = {
                    'true_answer': true_answer,
                    'original_answer': original_answer,
                    'correct_trials': correct_trials,
                    'total_trials': len(question_data),
                    'correct_percentage': (len(correct_trials) / len(question_data)) * 100
                }
    
    return examples

# Analyze the data
wrong_to_correct_examples = find_wrong_to_correct_examples(data)

if not wrong_to_correct_examples:
    print("No examples found where original was wrong but paragraph removal led to correct answers.")
else:
    # Prepare data for plotting
    question_ids = sorted(list(wrong_to_correct_examples.keys()), key=int)
    correct_counts = [len(wrong_to_correct_examples[qid]['correct_trials']) for qid in question_ids]
    total_counts = [wrong_to_correct_examples[qid]['total_trials'] for qid in question_ids]
    percentages = [wrong_to_correct_examples[qid]['correct_percentage'] for qid in question_ids]
    
    # Create figure with a bar chart and a heatmap for paragraph keeps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Bar chart for counts and percentages
    x = np.arange(len(question_ids))
    width = 0.35
    
    ax1.bar(x, total_counts, width, label='Total Trials', color='lightgray')
    ax1.bar(x, correct_counts, width, label='Correct After Removal', color='green')
    
    # Increase the spacing and adjust rotation to avoid overlapping with title
    for i, (total, correct) in enumerate(zip(total_counts, correct_counts)):
        percentage = (correct / total) * 100
        ax1.text(i, total + 0.7, f"{percentage:.1f}%", ha='center', fontsize=9, rotation=45)
    
    ax1.set_xlabel('Question ID')
    ax1.set_ylabel('Count')
    ax1.set_title('Wrong Original → Correct After Paragraph Removal')
    ax1.set_xticks(x)
    ax1.set_xticklabels(question_ids)
    ax1.legend()
    
    # Create a detailed heatmap showing paragraph keeps that led to correct answers
    all_paragraph_keeps = sorted(set(
        trial['paragraph_keeps'] 
        for examples in wrong_to_correct_examples.values() 
        for trial in examples['correct_trials']
    ))
    
    # Get all possible paragraph keeps across all questions
    all_possible_keeps = sorted(set(
        trial[2] for qid in question_ids for trial in data[int(qid)]
    ))
    
    # Initialize matrix for enhanced heatmap (2 = correct, 1 = same wrong answer as original, 0 = different wrong answer)
    heatmap_data = np.zeros((len(question_ids), len(all_possible_keeps)))
    
    # Fill in heatmap data
    for i, qid in enumerate(question_ids):
        qid_int = int(qid)
        example = wrong_to_correct_examples[qid]
        true_answer = example['true_answer']
        original_answer = example['original_answer']
        
        # Process each possible paragraph keep count
        for j, keep_count in enumerate(all_possible_keeps):
            # Find the trial with this keep count
            matching_trials = [t for t in data[qid_int] if t[2] == keep_count]
            
            if matching_trials:
                trial = matching_trials[0]
                removed_answer = normalize_answer(trial[3])
                
                if removed_answer == true_answer:
                    heatmap_data[i, j] = 2  # Correct answer (green)
                elif removed_answer == original_answer:
                    heatmap_data[i, j] = 1  # Same wrong answer as original (yellow)
                else:
                    heatmap_data[i, j] = -1  # Different wrong answer (red)
    
    # Create a custom colormap: red for different wrong answers, yellow for same wrong answers, green for correct answers
    cmap = plt.cm.colors.ListedColormap(['#FF9999', '#FFFFFF', '#FFFF99', '#99FF99'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot enhanced heatmap
    im = ax2.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')
    
    # Set ticks
    ax2.set_xticks(np.arange(len(all_possible_keeps)))
    ax2.set_yticks(np.arange(len(question_ids)))
    ax2.set_xticklabels(all_possible_keeps)
    ax2.set_yticklabels(question_ids)
    
    # Labels
    ax2.set_title('Effect of Paragraph Removal on Answer Correctness')
    ax2.set_xlabel('Number of Paragraphs Kept')
    ax2.set_ylabel('Question ID')
    
    # Add more vertical space for the legend
    fig.subplots_adjust(bottom=0.2)
    
    # Add text annotations and a legend
    for i in range(len(question_ids)):
        for j in range(len(all_possible_keeps)):
            if heatmap_data[i, j] == 2:
                ax2.text(j, i, '✓', ha='center', va='center', color='darkgreen', fontweight='bold')
            elif heatmap_data[i, j] == 1:
                ax2.text(j, i, '○', ha='center', va='center', color='goldenrod')
            elif heatmap_data[i, j] == -1:
                ax2.text(j, i, '✗', ha='center', va='center', color='darkred')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#99FF99', edgecolor='black', label='Correct answer'),
        Patch(facecolor='#FFFF99', edgecolor='black', label='Same wrong answer as original'),
        Patch(facecolor='#FF9999', edgecolor='black', label='Different wrong answer')
    ]
    ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    
    # Add more padding at the top to prevent overlap with titles
    plt.tight_layout(pad=2.0)
    # Set y-limit for the first axis to create more space for percentage labels
    ax1.set_ylim(0, max(total_counts) * 1.2)
    plt.show()
    fig.savefig("figures/wrong_org.png", dpi=600)

    # Print detailed analysis
    print("\nDetailed analysis of questions where wrong answers became correct after paragraph removal:")
    for qid in question_ids:
        example = wrong_to_correct_examples[qid]
        print(f"\nQuestion {qid}:")
        print(f"  True answer: {example['true_answer']}")
        print(f"  Original wrong answer: {example['original_answer']}")
        print(f"  {len(example['correct_trials'])} out of {example['total_trials']} trials resulted in correct answers ({example['correct_percentage']:.1f}%)")
        
        print("  Paragraph keeps that led to correct answers:")
        for trial in sorted(example['correct_trials'], key=lambda x: x['paragraph_keeps']):
            print(f"    - {trial['paragraph_keeps']} paragraph(s): Got correct answer {trial['answer']}")

# %%

data = answer_dict_reload

# Function to analyze the data for each question
def normalize_answer(answer):
    if answer is None:
        return None
        
    try:
        # Try to convert to float first (handles decimal strings)
        float_value = float(answer)
        
        # Check if it's an integer value
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except (ValueError, TypeError):
        # If conversion fails, return the original answer
        return answer

# Function to normalize answer values for comparison
def normalize_answer(answer):
    if answer is None:
        return None
        
    try:
        # Try to convert to float first (handles decimal strings)
        float_value = float(answer)
        
        # Check if it's an integer value
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except (ValueError, TypeError):
        # If conversion fails, return the original answer
        return answer

# Function to analyze the data for each question
def analyze_question(question_data):
    raw_true_answer = question_data[0][0]  # The true answer is the first element of the first trial
    raw_original_answer = question_data[0][1]  # The original tested answer
    
    # Normalize answers for comparison
    true_answer = normalize_answer(raw_true_answer)
    original_tested_answer = normalize_answer(raw_original_answer)
    
    # Check if the original tested answer was correct
    original_correct = original_tested_answer == true_answer
    
    if not original_correct:
        return 0, 0  # If original wasn't correct, return 0
    
    # Count correct answers after paragraph removal with normalized comparison
    correct_after_removal = 0
    for trial in question_data:
        normalized_trial_answer = normalize_answer(trial[3])
        if normalized_trial_answer == true_answer:
            correct_after_removal += 1
            
    total_trials = len(question_data)
    
    return correct_after_removal, total_trials

# Analyze each question
results = {}
for question_id, question_data in data.items():
    correct_after_removal, total_trials = analyze_question(question_data)
    
    # Only include questions where the original answer was correct
    if total_trials > 0:
        results[question_id] = {
            'correct_after_removal': correct_after_removal,
            'total_trials': total_trials,
            'percentage': (correct_after_removal / total_trials) * 100
        }

# Prepare data for plotting
question_ids = sorted(list(results.keys()), key=int)  # Sort numerically
correct_counts = [results[qid]['correct_after_removal'] for qid in question_ids]
total_counts = [results[qid]['total_trials'] for qid in question_ids]
percentages = [results[qid]['percentage'] for qid in question_ids]

# Create figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(23, 7))

# Bar plot 1: Counts
x = np.arange(len(question_ids))
width = 0.35

ax1.bar(x, total_counts, width, label='Total Trials')
ax1.bar(x, correct_counts, width, label='Correct After Removal')
ax1.set_xlabel('Question ID')
ax1.set_ylabel('Count')
ax1.set_title('Correct Answers After Paragraph Removal')
ax1.set_xticks(x)
ax1.set_xticklabels(question_ids)
ax1.legend()

# Bar plot 2: Percentages
bar_width = 0.6  # Making bars narrower to add spacing
ax2.bar(x, percentages, bar_width, color='green')
ax2.set_xlabel('Question ID')
ax2.set_ylabel('Percentage (%)')
ax2.set_title('Percentage of Correct Answers After Paragraph Removal')
ax2.set_xticks(x)
ax2.set_xticklabels(question_ids)
ax2.set_ylim(0, 115)  # Increased the upper limit to make room for labels

# Improve label positioning to avoid overlap
for i, percentage in enumerate(percentages):
    ax2.text(i, percentage + 2, f'{percentage:.1f}%', 
             ha='center', fontsize=9, rotation=45)

# plt.tight_layout()
# plt.show()


# Print the results for reference
# for qid in question_ids:
#     print(f"Question {qid}: {results[qid]['correct_after_removal']} correct out of {results[qid]['total_trials']} trials ({results[qid]['percentage']:.1f}%)")
    
# Function to normalize answer values for comparison
def normalize_answer(answer):
    if answer is None:
        return None
        
    try:
        # Try to convert to float first (handles decimal strings)
        float_value = float(answer)
        
        # Check if it's an integer value
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except (ValueError, TypeError):
        # If conversion fails, return the original answer
        return answer

# Function to find examples where original was correct and analyze paragraph removal effects
def find_correct_original_examples(data):
    examples = {}
    
    for question_id, question_data in data.items():
        true_answer = normalize_answer(question_data[0][0])  # The true answer
        original_answer = normalize_answer(question_data[0][1])  # The original tested answer
        
        # Check if the original answer was correct
        if true_answer == original_answer:
            # Analyze how paragraph removal affects correctness
            total_trials = len(question_data)
            correct_after_removal = []
            all_trials = []
            
            for trial in question_data:
                paragraph_keeps = trial[2]  # Number of paragraph keeps
                removed_answer = normalize_answer(trial[3])  # Answer after removal
                
                all_trials.append({
                    'paragraph_keeps': paragraph_keeps,
                    'answer': removed_answer,
                    'is_correct': removed_answer == true_answer
                })
                
                if removed_answer == true_answer:
                    correct_after_removal.append({
                        'paragraph_keeps': paragraph_keeps,
                        'answer': removed_answer
                    })
            
            examples[question_id] = {
                'true_answer': true_answer,
                'original_answer': original_answer,
                'correct_trials': correct_after_removal,
                'all_trials': all_trials,
                'total_trials': total_trials,
                'correct_percentage': (len(correct_after_removal) / total_trials) * 100
            }
    
    return examples

# Analyze the data
correct_original_examples = find_correct_original_examples(data)

if not correct_original_examples:
    print("No examples found where original was correct.")
else:
    # Prepare data for plotting
    question_ids = sorted(list(correct_original_examples.keys()), key=int)
    correct_counts = [len(correct_original_examples[qid]['correct_trials']) for qid in question_ids]
    total_counts = [correct_original_examples[qid]['total_trials'] for qid in question_ids]
    percentages = [correct_original_examples[qid]['correct_percentage'] for qid in question_ids]
    
    # Create figure with a bar chart and a heatmap
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # # Bar chart for counts and percentages
    # x = np.arange(len(question_ids))
    # width = 0.35
    
    # ax1.bar(x, total_counts, width, label='Total Trials', color='lightgray')
    # ax1.bar(x, correct_counts, width, label='Correct After Removal', color='green')
    
    # # Increase the spacing and adjust rotation to avoid overlapping with title
    # for i, (total, correct) in enumerate(zip(total_counts, correct_counts)):
    #     percentage = (correct / total) * 100
    #     ax1.text(i, total + 0.7, f"{percentage:.1f}%", ha='center', fontsize=9, rotation=45)
    
    # ax1.set_xlabel('Question ID')
    # ax1.set_ylabel('Count')
    # ax1.set_title('Correct Original → Effect After Paragraph Removal')
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(question_ids)
    # ax1.legend()
    # # Set y-limit for the first axis to create more space for percentage labels
    # ax1.set_ylim(0, max(total_counts) * 1.2)
    
    # Get all possible paragraph keeps across all questions
    all_possible_keeps = sorted(set(
        trial[2] for qid in question_ids for trial in data[int(qid)]
    ))
    
    # Initialize matrix for enhanced heatmap (1 = correct, 0 = wrong/missing answer)
    heatmap_data = np.zeros((len(question_ids), len(all_possible_keeps)))
    
    # Fill in heatmap data
    for i, qid in enumerate(question_ids):
        qid_int = int(qid)
        example = correct_original_examples[qid]
        true_answer = example['true_answer']
        
        # Process each possible paragraph keep count
        for j, keep_count in enumerate(all_possible_keeps):
            # Find the trial with this keep count
            matching_trials = [t for t in data[qid_int] if t[2] == keep_count]
            
            if matching_trials:
                trial = matching_trials[0]
                removed_answer = normalize_answer(trial[3])
                
                if removed_answer == true_answer:
                    heatmap_data[i, j] = 1  # Correct answer (green)
                elif removed_answer is None:
                    heatmap_data[i, j] = -1  # No answer found (gray)
                else:
                    heatmap_data[i, j] = 0  # Wrong answer (red)
    
    # Create a custom colormap
    cmap = plt.cm.colors.ListedColormap(['#FF9999', '#DDDDDD', '#99FF99'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot enhanced heatmap
    im = ax3.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')
    
    # Add text annotations and a legend
    for i in range(len(question_ids)):
        for j in range(len(all_possible_keeps)):
            if heatmap_data[i, j] == 1:
                ax3.text(j, i, '✓', ha='center', va='center', color='darkgreen', fontweight='bold')
            elif heatmap_data[i, j] == -1:
                ax3.text(j, i, '∅', ha='center', va='center', color='dimgray')
            elif heatmap_data[i, j] == 0:
                ax3.text(j, i, '✗', ha='center', va='center', color='darkred')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#99FF99', edgecolor='black', label='Maintained correct answer'),
        Patch(facecolor='#FF9999', edgecolor='black', label='Changed to wrong answer'),
        Patch(facecolor='#DDDDDD', edgecolor='black', label='No answer found')
    ]
    ax3.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    
    # Set ticks
    ax3.set_xticks(np.arange(len(all_possible_keeps)))
    ax3.set_yticks(np.arange(len(question_ids)))
    ax3.set_xticklabels(all_possible_keeps)
    ax3.set_yticklabels(question_ids)
    
    # Labels
    ax3.set_title('Effect of Paragraph Removal on Originally Correct Answers')
    ax3.set_xlabel('Number of Paragraphs Kept')
    ax3.set_ylabel('Question ID')
    
    # Add more vertical space for the legend
    fig.subplots_adjust(bottom=0.2)
    
    # Add more padding at the top to prevent overlap with titles
    plt.tight_layout(pad=2.0)
    
    plt.show()

    fig.savefig("figures/correct_org2.png", dpi=600)
