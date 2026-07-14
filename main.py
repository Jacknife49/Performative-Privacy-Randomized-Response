import os
from heatmap_generator import generate_heatmap_for_model
from population_dynamics import plot_population_history_only

def main():
    # Ensure plots directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # ==============================================================
    # 1. Generate Heatmap for a chosen model
    # ==============================================================
    chosen_model = "agnostic"  # Toggle this to "original" if needed
    
    print("-" * 50)
    print(f"TASK 1: Heatmap Generation ({chosen_model.capitalize()} Model)")
    print("-" * 50)
    generate_heatmap_for_model(model_type=chosen_model)


    # ==============================================================
    # 2. Output 3 specific population evolutions 
    # ==============================================================
    print("\n" + "-" * 50)
    print("TASK 2: Population Evolutions (q vs r conditions)")
    print("-" * 50)

    # Condition A: q < r (Leave rate is less than join rate -> Pop should grow)
    plot_population_history_only(
        leave_rate=0.2, 
        join_rate=0.6, 
        growth_model="original",
        output_filename=os.path.join(output_dir, "pop_history_q_less_than_r.pdf")
    )

    # Condition B: q > r (Leave rate is greater than join rate -> Pop should shrink)
    plot_population_history_only(
        leave_rate=0.6, 
        join_rate=0.2, 
        growth_model="original",
        output_filename=os.path.join(output_dir, "pop_history_q_greater_than_r.pdf")
    )

    # Condition C: q = r (Rates are equal -> Pop should stabilize or hover)
    plot_population_history_only(
        leave_rate=0.4, 
        join_rate=0.4, 
        growth_model="original",
        output_filename=os.path.join(output_dir, "pop_history_q_equals_r.pdf")
    )

    print("\nAll tasks complete! Check the 'plots' folder for your generated PDFs.")

if __name__ == "__main__":
    main()
