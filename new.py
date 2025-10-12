import numpy

def greedy_tsp(distance_matrix, start_node=0):
    num_nodes = len(distance_matrix)
    path = [start_node]
    visited = {start_node}
    current_node = start_node
    total_distance = 0

    while len(visited) < num_nodes:
        min_dist = numpy.inf
        next_node = -1
        for neighbor in range(num_nodes):
            if neighbor not in visited:
                dist = distance_matrix[current_node][neighbor]
                if dist < min_dist:
                    min_dist = dist
                    next_node = neighbor
        if next_node != -1:
            path.append(next_node)
            visited.add(next_node)
            total_distance += min_dist
            current_node = next_node
        else:
            return numpy.inf, []

    if num_nodes > 1:
        total_distance += distance_matrix[current_node][start_node]
        path.append(start_node)

    return total_distance, path

def main():
    print("Greedy TSP Solver (Terminal Edition)")
    N = int(input("Enter number of stops (N): "))
    print(f"Enter the {N}x{N} distance matrix row-wise (space separated):\n")
    distance_matrix = []
    for i in range(N):
        row = list(map(int, input(f"Row {i+1}: ").strip().split()))
        if len(row) != N:
            print(f"Error: Row must have {N} values.")
            return
        distance_matrix.append(row)
    distance_matrix = numpy.array(distance_matrix)

    print("\nCalculating heuristic route...")
    total_distance, path = greedy_tsp(distance_matrix, start_node=0)
    node_labels = [chr(65 + i) for i in range(N)]
    if path:
        path_labels = [node_labels[i] for i in path]
        print("\nHeuristic Route Found (Nearest Neighbor):")
        print("Route Distance:", total_distance)
        print("Calculated Path:", " -> ".join(path_labels))
    else:
        print("\nNo valid tour found. Please check your distance matrix.")

if __name__ == "__main__":
    main()
