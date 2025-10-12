import numpy
import heapq # Essential for the priority queue used in Branch & Bound

# =========================================================================
# === Branch & Bound TSP Core Functions (Unchanged) ===
# =========================================================================

def calculate_cost_so_far(partial_path, dist_matrix):
    """Calculates the exact cost of the path traveled so far."""
    cost = 0
    for i in range(len(partial_path) - 1):
        u, v = partial_path[i], partial_path[i+1]
        cost += dist_matrix[u][v]
    return cost

def get_lower_bound(partial_path, dist_matrix):
    """Calculates a simplified lower bound for the total tour cost."""
    num_nodes = len(dist_matrix)
    if len(partial_path) == 0: return 0
    
    cost_so_far = calculate_cost_so_far(partial_path, dist_matrix)
    visited = set(partial_path)
    min_future_cost = 0
    unvisited_nodes = set(range(num_nodes)) - visited
    
    for node in unvisited_nodes:
        min_edge = numpy.inf
        for neighbor in range(num_nodes):
            if node != neighbor and dist_matrix[node][neighbor] < min_edge:
                min_edge = dist_matrix[node][neighbor]
        
        if min_edge != numpy.inf:
            min_future_cost += min_edge
            
    return cost_so_far + min_future_cost

def branch_and_bound_tsp(distance_matrix, start_node=0):
    """Implements the Branch and Bound algorithm to find the optimal TSP tour."""
    num_nodes = len(distance_matrix)
    if num_nodes <= 1: return 0, [start_node, start_node] if num_nodes == 1 else []

    pq = [(get_lower_bound([start_node], distance_matrix), 
           [start_node], 
           start_node, 
           frozenset(range(num_nodes)) - {start_node})]
    
    min_cost = numpy.inf
    best_path = []

    while pq:
        lower_bound, current_path, current_node, unvisited = heapq.heappop(pq)

        if lower_bound >= min_cost: continue

        if not unvisited:
            cost_to_close = distance_matrix[current_node][start_node]
            cost = calculate_cost_so_far(current_path, distance_matrix) + cost_to_close
            
            if cost < min_cost:
                min_cost = cost
                best_path = current_path + [start_node]
            continue

        for next_node in unvisited:
            if distance_matrix[current_node][next_node] == 0 and current_node != next_node:
                continue

            new_path = current_path + [next_node]
            new_unvisited = unvisited - {next_node}
            new_bound = get_lower_bound(new_path, distance_matrix)
            
            if new_bound < min_cost:
                heapq.heappush(pq, (new_bound, new_path, next_node, new_unvisited))

    return min_cost, best_path

# =========================================================================
# === Dynamic Stop Insertion Function (Using Optimal Insertion Heuristic) ===
# =========================================================================

def insert_stop_in_path(existing_distance_matrix, existing_path, new_stop_index):
    """
    Finds the optimal position to insert a new_stop into an existing_path.
    This is a fast O(N) operation to check all existing edges (u -> v) and 
    find the one that minimizes (u -> new_stop -> v) - (u -> v).
    """
    if not existing_path:
        return numpy.inf, []

    # Path without the final return-to-start node
    tour_nodes = existing_path[:-1] 
    
    best_insertion_cost_increase = numpy.inf
    best_insertion_index = -1
    
    N = len(tour_nodes)
    
    # Iterate through all N edges in the circuit
    for i in range(N):
        u = tour_nodes[i]
        v = tour_nodes[(i + 1) % N] # Wrap around for the last edge back to start
        
        # Original cost of the edge
        original_cost = existing_distance_matrix[u][v]
        
        # New path cost: u -> new_stop -> v
        cost_u_to_new = existing_distance_matrix[u][new_stop_index]
        cost_new_to_v = existing_distance_matrix[new_stop_index][v]
        
        new_path_cost_segment = cost_u_to_new + cost_new_to_v
        
        # Cost increase for this specific insertion
        increase = new_path_cost_segment - original_cost
        
        if increase < best_insertion_cost_increase:
            best_insertion_cost_increase = increase
            # Index i + 1 means the stop is inserted *after* tour_nodes[i]
            best_insertion_index = i + 1 

    if best_insertion_index == -1 or best_insertion_cost_increase == numpy.inf:
         return numpy.inf, []

    # Construct the new path
    new_tour_nodes = tour_nodes[:best_insertion_index] + [new_stop_index] + tour_nodes[best_insertion_index:]
    
    # Recalculate total distance
    new_path = new_tour_nodes + [new_tour_nodes[0]]
    new_total_distance = calculate_cost_so_far(new_path, existing_distance_matrix)
    
    return new_total_distance, new_path


def get_new_stop_distances(N_current, new_stop_label):
    """Handles the terminal input for the new stop's distances."""
    print(f"\n--- Input for New Stop '{new_stop_label}' (Node {N_current}) ---")
    print(f"Enter {N_current} space-separated distances (A to {new_stop_label}, B to {new_stop_label}, etc.):")
    
    try:
        dist_row = list(map(int, input("New Distances: ").strip().split()))
    except ValueError:
        print("Error: Distances must be integers. Insertion failed.")
        return None

    if len(dist_row) != N_current:
        print(f"Error: Must provide exactly {N_current} distances. Insertion failed.")
        return None
        
    if any(d < 0 for d in dist_row):
        print("Error: Distances cannot be negative. Insertion failed.")
        return None

    return numpy.array(dist_row)


# =========================================================================
# === Terminal Interface (Modified Main function) ===
# =========================================================================

def main():
    print("Branch and Bound TSP Solver (Terminal Edition)")
    
    # --- 1. Initial Setup and Solving ---
    try:
        N = int(input("Enter initial number of stops (N, e.g., 4): "))
    except ValueError:
        print("Invalid input for N. Exiting.")
        return

    if N < 2:
        print("Must have at least 2 stops.")
        return

    print(f"Enter the {N}x{N} distance matrix row-wise (space separated):\n")
    distance_matrix = []
    
    for i in range(N):
        try:
            row = list(map(int, input(f"Row {chr(65+i)}: ").strip().split()))
        except ValueError:
            print("Error: Distances must be integers. Exiting.")
            return

        if len(row) != N:
            print(f"Error: Row must have exactly {N} values. Exiting.")
            return
        distance_matrix.append(row)
        
    D_matrix = numpy.array(distance_matrix)
    
    print("\nCalculating optimal route (Branch & Bound)...")
    total_distance, path = branch_and_bound_tsp(D_matrix, start_node=0)
    
    N_current = N
    D_matrix_current = D_matrix
    
    # --- Initial Output ---
    node_labels = [chr(65 + i) for i in range(N_current)]
    print("\n-------------------------------------------")
    if path and total_distance != numpy.inf:
        path_labels = [node_labels[i] for i in path]
        print("🎉 INITIAL OPTIMAL ROUTE FOUND (B&B) 🎉")
        print("Route Distance:", total_distance)
        print("Calculated Path:", " -> ".join(path_labels))
    else:
        print("❌ ERROR: No valid tour found. Cannot proceed to dynamic insertion.")
        print("-------------------------------------------")
        return
    print("-------------------------------------------")

    # --- 2. Dynamic Insertion Loop ---
    while True:
        choice = input("\nDo you want to insert a new stop? (y/n): ").lower().strip()
        if choice != 'y':
            break

        # Calculate the new stop's index and label
        new_stop_index = N_current
        new_stop_label = chr(65 + new_stop_index)
        
        # Get distances for the new stop
        new_dist_input = get_new_stop_distances(N_current, new_stop_label)
        if new_dist_input is None:
            continue

        # --- Augment the Distance Matrix (N+1 x N+1) ---
        
        # 1. Extend the existing N x N matrix to N x (N+1) (add new column)
        D_matrix_extended = numpy.pad(D_matrix_current, ((0, 0), (0, 1)), 'constant', constant_values=0)
        D_matrix_extended[:, new_stop_index] = new_dist_input # Distances A->New, B->New, ...

        # 2. Add the new row for New->A, New->B, ... and set New->New=0
        new_row = new_dist_input.tolist() + [0]
        D_matrix_extended = numpy.vstack([D_matrix_extended, new_row])

        # --- Run the Insertion Logic ---
        
        new_distance, new_path = insert_stop_in_path(D_matrix_extended, path, new_stop_index)
        
        # --- Update and Output Results ---
        
        if new_path and new_distance != numpy.inf:
            N_current += 1
            D_matrix_current = D_matrix_extended # Update the current matrix for the next iteration
            path = new_path
            total_distance = new_distance
            
            new_node_labels = [chr(65 + i) for i in range(N_current)]
            new_path_labels = [new_node_labels[i] for i in new_path]
            
            print("\n===========================================")
            print(f"✅ STOP '{new_stop_label}' INSERTED OPTIMALLY! (Fast Heuristic)")
            print("===========================================")
            print(f"New Total Stops: {N_current} (A to {new_stop_label})")
            print("New Optimal Distance:", total_distance)
            print("New Calculated Path:", " -> ".join(new_path_labels))
            print("===========================================")
        else:
            print("\n❌ Insertion failed due to unreachable path after adding the new stop.")


if __name__ == "__main__":
    main()