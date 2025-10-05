import numpy
import heapq
import tkinter as tk
from tkinter import messagebox, scrolledtext

# =========================================================================
# === 1. GREEDY (NEAREST NEIGHBOR) TSP FUNCTION ===
# =========================================================================

def greedy_tsp(distance_matrix, start_node=0):
    """
    Implements the Nearest Neighbor (Greedy) TSP Heuristic.
    Finds a fast, sub-optimal route by always choosing the closest unvisited node.
    Returns (total_distance, path)
    """
    num_nodes = len(distance_matrix)
    path = [start_node]
    visited = {start_node}
    current_node = start_node
    total_distance = 0

    # 1. Build the path greedily
    while len(visited) < num_nodes:
        min_dist = numpy.inf
        next_node = -1
        
        # Find the nearest unvisited neighbor
        for neighbor in range(num_nodes):
            if neighbor not in visited:
                dist = distance_matrix[current_node][neighbor]
                if dist < min_dist:
                    min_dist = dist
                    next_node = neighbor
        
        # Move to the nearest neighbor
        if next_node != -1:
            path.append(next_node)
            visited.add(next_node)
            total_distance += min_dist
            current_node = next_node
        else:
            # Should not happen in a complete graph, but good for safety
            return numpy.inf, [] 

    # 2. Complete the circuit by returning to the start node
    if num_nodes > 1:
        total_distance += distance_matrix[current_node][start_node]
        path.append(start_node)
    
    return total_distance, path


# =========================================================================
# === 2. BRANCH & BOUND TSP FUNCTIONS (Required for comparison) ===
# (Functions remain the same as the previous correct code)
# =========================================================================

def calculate_cost_so_far(partial_path, dist_matrix):
    cost = 0
    for i in range(len(partial_path) - 1):
        u, v = partial_path[i], partial_path[i+1]
        cost += dist_matrix[u][v]
    return cost

def get_lower_bound(partial_path, dist_matrix):
    if len(partial_path) <= 1: return 0
    cost_so_far = calculate_cost_so_far(partial_path, dist_matrix)
    num_nodes = len(dist_matrix)
    visited = set(partial_path)
    min_unvisited_cost = 0
    unvisited_nodes = set(range(num_nodes)) - visited
    for node in unvisited_nodes:
        min_edge = numpy.inf
        for neighbor in range(num_nodes):
            if node != neighbor and dist_matrix[node][neighbor] < min_edge:
                min_edge = dist_matrix[node][neighbor]
        if min_edge != numpy.inf: min_unvisited_cost += min_edge
    return cost_so_far + min_unvisited_cost

def branch_and_bound_tsp(distance_matrix, start_node=0):
    num_nodes = len(distance_matrix)
    pq = [(get_lower_bound([start_node], distance_matrix), [start_node], start_node, set(range(num_nodes)) - {start_node})]
    min_cost = numpy.inf
    best_path = []
    while pq:
        lower_bound, current_path, current_node, unvisited = heapq.heappop(pq)
        if lower_bound >= min_cost: continue
        if not unvisited:
            cost = calculate_cost_so_far(current_path, distance_matrix) + distance_matrix[current_node][start_node]
            if cost < min_cost:
                min_cost = cost
                best_path = current_path + [start_node]
            continue
        for next_node in unvisited:
            if distance_matrix[current_node][next_node] == 0 and current_node != next_node: continue
            new_path = current_path + [next_node]
            new_unvisited = unvisited - {next_node}
            new_bound = get_lower_bound(new_path, distance_matrix) 
            if new_bound < min_cost:
                heapq.heappush(pq, (new_bound, new_path, next_node, new_unvisited))
    return min_cost, best_path

# =========================================================================
# === 3. TKINTER GUI IMPLEMENTATION (Updated for both solvers) ===
# =========================================================================

class TSP_Solver_UI:
    def __init__(self, master):
        self.master = master
        master.title("🚌 College Bus Route Planner (TSP)")
        master.geometry("520x620") 
        
        # --- Variables & Styling ---
        self.num_stops = tk.IntVar(value=4)
        self.entry_matrix = None 
        bg_color = "#f5f5f5"
        master.configure(bg=bg_color)
        
        # --- Top Frame for Stop Count ---
        self.count_frame = tk.Frame(master, bg=bg_color)
        self.count_frame.pack(pady=10)
        
        tk.Label(self.count_frame, text="Number of Stops (N):", bg=bg_color, font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.stop_entry = tk.Entry(self.count_frame, textvariable=self.num_stops, width=5, font=('Arial', 10), justify='center', bd=1, relief=tk.SOLID)
        self.stop_entry.pack(side=tk.LEFT, padx=5)
        self.stop_entry.bind('<Return>', self.create_matrix_grid)
        
        tk.Button(self.count_frame, text="Generate Matrix", command=self.create_matrix_grid, bg="#6c757d", fg="white", font=('Arial', 10, 'bold'), bd=0, activebackground="#5a6268").pack(side=tk.LEFT, padx=10)

        # --- Matrix Input Frame ---
        self.matrix_frame = tk.Frame(master, bg=bg_color)
        self.matrix_frame.pack(pady=10, padx=10)

        # --- Solve Buttons Frame ---
        self.btn_frame = tk.Frame(master, bg=bg_color)
        self.btn_frame.pack(pady=5)
        
        # New: Greedy Solver Button (Heuristic - Fast)
        self.greedy_button = tk.Button(self.btn_frame, text="1. Solve (Greedy / Heuristic)", command=lambda: self.solve('greedy'), 
                                       bg="#FFC107", fg="black", font=('Arial', 10, 'bold'), bd=0, activebackground="#e0a800", state=tk.DISABLED)
        self.greedy_button.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)

        # Branch & Bound Solver Button (Exact - Optimal)
        self.bb_button = tk.Button(self.btn_frame, text="2. Solve (B&B / Optimal)", command=lambda: self.solve('bnb'), 
                                   bg="#007BFF", fg="white", font=('Arial', 10, 'bold'), bd=0, activebackground="#0056b3", state=tk.DISABLED)
        self.bb_button.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)


        # --- Aesthetic Output Display ---
        tk.Label(master, text="Solution Results:", bg=bg_color, font=('Arial', 11, 'bold')).pack(pady=(5, 2))
        
        self.output_frame = tk.Frame(master, bd=1, relief=tk.SOLID, padx=5, pady=5, bg="#ffffff")
        self.output_frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(self.output_frame, height=12, width=50, state=tk.DISABLED, wrap=tk.WORD, 
                                   font=('Consolas', 11), bg="#ffffff", bd=0) 
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Define Aesthetic Tags
        self.output_text.tag_configure("success", foreground="#155724", background="#d4edda", font=('Consolas', 11, 'bold'))
        self.output_text.tag_configure("heuristic", foreground="#856404", background="#fff3cd", font=('Consolas', 11, 'bold')) # Yellow for Greedy
        self.output_text.tag_configure("error", foreground="#721c24", background="#f8d7da", font=('Consolas', 11, 'bold'))   
        self.output_text.tag_configure("result", font=('Consolas', 11))
        self.output_text.tag_configure("distance", foreground="#007BFF", font=('Consolas', 12, 'bold'))
        self.output_text.tag_configure("distance_g", foreground="#FFC107", font=('Consolas', 12, 'bold')) # Yellow for Greedy distance

        self.create_matrix_grid()

    def create_matrix_grid(self, event=None):
        """Creates the input fields for the distance matrix and enables buttons."""
        # ... [Matrix creation logic remains the same] ...
        try:
            N = self.num_stops.get()
            if N < 3 or N > 8:
                messagebox.showerror("Error", "Please enter N between 3 and 8 for a quick result.")
                self.num_stops.set(4)
                return
        except tk.TclError:
            messagebox.showerror("Error", "Invalid number of stops.")
            return

        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        self.entry_matrix = []
        labels = [chr(65 + i) for i in range(N)] 

        # Header/Input Row creation (simplified for brevity, assumes standard grid logic)
        tk.Label(self.matrix_frame, text="From \\ To", bg="#e9ecef", font=('Arial', 9, 'bold'), width=5, bd=1, relief=tk.RIDGE).grid(row=0, column=0, padx=1, pady=1)
        for j in range(N):
            tk.Label(self.matrix_frame, text=labels[j], bg="#e9ecef", font=('Arial', 9, 'bold'), width=5, bd=1, relief=tk.RIDGE).grid(row=0, column=j+1, padx=1, pady=1)

        for i in range(N):
            row_entries = []
            tk.Label(self.matrix_frame, text=labels[i], bg="#e9ecef", font=('Arial', 9, 'bold'), width=5, bd=1, relief=tk.RIDGE).grid(row=i+1, column=0, padx=1, pady=1)
            for j in range(N):
                entry = tk.Entry(self.matrix_frame, width=5, justify='center', bd=1, relief=tk.GROOVE)
                entry.grid(row=i+1, column=j+1, padx=1, pady=1)
                if i == j:
                    entry.insert(0, "0")
                    entry.config(state='readonly', disabledbackground="#f8f9fa")
                row_entries.append(entry)
            self.entry_matrix.append(row_entries)
        # --- End of Matrix creation logic ---

        self.greedy_button.config(state=tk.NORMAL)
        self.bb_button.config(state=tk.NORMAL)
        self.update_output("Matrix ready. Choose a solver option above.", tag="result")

    def get_matrix_from_entries(self, N):
        """Extracts the distance matrix from the Entry widgets."""
        matrix = []
        for i in range(N):
            row = []
            for j in range(N):
                try:
                    val = self.entry_matrix[i][j].get()
                    if not val: raise ValueError
                    row.append(int(val))
                except ValueError:
                    messagebox.showerror("Input Error", f"Distance from {chr(65+i)} to {chr(65+j)} is invalid or empty.")
                    return None
            matrix.append(row)
        return numpy.array(matrix)

    def update_output(self, text, tag="result"):
        """Updates the Text widget with styled results."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text, tag)
        self.output_text.config(state=tk.DISABLED)
    
    def solve(self, method):
        """Handles the solve button click for either Greedy or B&B."""
        N = self.num_stops.get()
        D_matrix = self.get_matrix_from_entries(N)
        if D_matrix is None: return

        if method == 'bnb':
            self.update_output("Calculating optimal (B&B)... Please wait.", tag="result")
            self.master.update() 
            min_distance, path = branch_and_bound_tsp(D_matrix, start_node=0)
            header_tag = "success"
            distance_tag = "distance"
            header_text = "🎉 OPTIMAL ROUTE FOUND (Branch & Bound)! 🎉\n"
        elif method == 'greedy':
            self.update_output("Calculating heuristic (Greedy)... Fast calculation.", tag="result")
            self.master.update() 
            min_distance, path = greedy_tsp(D_matrix, start_node=0)
            header_tag = "heuristic"
            distance_tag = "distance_g"
            header_text = "💡 HEURISTIC ROUTE FOUND (Greedy / Nearest Neighbor) 💡\n"
        else:
            return

        try:
            # --- Format Aesthetic Output ---
            node_labels = [chr(65 + i) for i in range(N)]
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)

            if path:
                path_labels = [node_labels[i] for i in path]
                
                # Header
                self.output_text.insert(tk.END, header_text, header_tag)
                self.output_text.insert(tk.END, "------------------------------------------\n\n", "result")
                
                # Distance Result
                self.output_text.insert(tk.END, "Route Distance: ", "result")
                self.output_text.insert(tk.END, f"{min_distance}\n\n", distance_tag)
                
                # Path Result
                self.output_text.insert(tk.END, "Calculated Path:\n", "result")
                self.output_text.insert(tk.END, f"  {' -> '.join(path_labels)}\n", "result")
                self.output_text.insert(tk.END, f"\nPath Length: {N + 1} stops (A->...->A)", "result")

            else:
                self.output_text.insert(tk.END, "❌ ERROR: No tour found.\n", "error")
                self.output_text.insert(tk.END, "The distance matrix may contain un-reachable stops.", "result")
            
            self.output_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Runtime Error", f"An unexpected error occurred: {e}")
            self.update_output(f"Error: {e}", tag="error")

# --- Main Tkinter Loop ---
if __name__ == "__main__":
    try:
        numpy.inf
    except NameError:
        print("Error: NumPy is required. Please install it using 'pip install numpy'")
        exit()

    root = tk.Tk()
    app = TSP_Solver_UI(root)
    root.mainloop()