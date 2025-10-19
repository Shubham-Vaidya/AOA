import numpy
import heapq
import tkinter as tk
from tkinter import messagebox

# =========================================================================
# === 1. GREEDY (NEAREST NEIGHBOR) TSP FUNCTION (UNCHANGED) ===
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
            return numpy.inf, [] 

    # 2. Complete the circuit by returning to the start node
    if num_nodes > 1:
        total_distance += distance_matrix[current_node][start_node]
        path.append(start_node)
    
    return total_distance, path


# =========================================================================
# === 2. BRANCH & BOUND TSP FUNCTIONS (UNCHANGED) ===
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
    # The first element is the lower bound for the priority queue
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
# === 3. TKINTER GUI IMPLEMENTATION (UPDATED) ===
# =========================================================================

class TSP_Solver_UI:
    def __init__(self, master):
        self.master = master
        master.title("🚌 College Bus Route Planner (TSP)")
        master.geometry("520x620") 
        
        # --- Variables & Styling ---
        self.num_stops = tk.IntVar(value=4)
        self.start_node_var = tk.StringVar(value='A') # NEW: Variable for selected start node
        self.entry_matrix = None # Stores the tk.Entry widgets
        self.current_matrix_data = [] # Stores the actual distance values for pre-filling
        bg_color = "#f5f5f5"
        master.configure(bg=bg_color)
        
        # --- Top Frame for Stop Count, Generate/Add, and Start Node Selection ---
        self.top_controls_frame = tk.Frame(master, bg=bg_color)
        self.top_controls_frame.pack(pady=10)
        
        # Row 1: Stop Count and Generate/Add Buttons
        self.count_frame = tk.Frame(self.top_controls_frame, bg=bg_color)
        self.count_frame.pack()
        
        tk.Label(self.count_frame, text="N Stops:", bg=bg_color, font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.stop_entry = tk.Entry(self.count_frame, textvariable=self.num_stops, width=5, font=('Arial', 10), justify='center', bd=1, relief=tk.SOLID)
        self.stop_entry.pack(side=tk.LEFT, padx=5)
        self.stop_entry.bind('<Return>', self.create_matrix_grid)
        
        tk.Button(self.count_frame, text="Generate Matrix", command=self.create_matrix_grid, bg="#6c757d", fg="white", font=('Arial', 10, 'bold'), bd=0, activebackground="#5a6268").pack(side=tk.LEFT, padx=(10, 5))
        tk.Button(self.count_frame, text="➕ Add New Stop", command=self.add_new_stop, bg="#28A745", fg="white", font=('Arial', 10, 'bold'), bd=0, activebackground="#1e7e34").pack(side=tk.LEFT, padx=5)

        # Row 2: Start Node Selection (NEW)
        self.start_node_frame = tk.Frame(self.top_controls_frame, bg=bg_color)
        self.start_node_frame.pack(pady=5)
        
        tk.Label(self.start_node_frame, text="Select Start Stop:", bg=bg_color, font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        self.start_node_menu = tk.OptionMenu(self.start_node_frame, self.start_node_var, 'A', *[]) # Placeholder options
        self.start_node_menu.config(bg="#dee2e6", font=('Arial', 10), activebackground="#c6c6c6")
        self.start_node_menu["menu"].config(font=('Arial', 10))
        self.start_node_menu.pack(side=tk.LEFT, padx=5)

        # --- Matrix Input Frame ---
        self.matrix_frame = tk.Frame(master, bg=bg_color)
        self.matrix_frame.pack(pady=10, padx=10)

        # --- Solve Buttons Frame ---
        self.btn_frame = tk.Frame(master, bg=bg_color)
        self.btn_frame.pack(pady=5)
        
        self.greedy_button = tk.Button(self.btn_frame, text="1. Solve (Greedy / Heuristic)", command=lambda: self.solve('greedy'), 
                                       bg="#FFC107", fg="black", font=('Arial', 10, 'bold'), bd=0, activebackground="#e0a800", state=tk.DISABLED)
        self.greedy_button.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)

        self.bb_button = tk.Button(self.btn_frame, text="2. Solve (B&B / Optimal)", command=lambda: self.solve('bnb'), 
                                   bg="#007BFF", fg="white", font=('Arial', 10, 'bold'), bd=0, activebackground="#0056b3", state=tk.DISABLED)
        self.bb_button.pack(side=tk.LEFT, padx=10, ipadx=10, ipady=5)


        # --- Aesthetic Output Display (UNCHANGED) ---
        tk.Label(master, text="Solution Results:", bg=bg_color, font=('Arial', 11, 'bold')).pack(pady=(5, 2))
        
        self.output_frame = tk.Frame(master, bd=1, relief=tk.SOLID, padx=5, pady=5, bg="#ffffff")
        self.output_frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(self.output_frame, height=12, width=50, state=tk.DISABLED, wrap=tk.WORD, 
                                   font=('Consolas', 11), bg="#ffffff", bd=0) 
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Define Aesthetic Tags
        self.output_text.tag_configure("success", foreground="#155724", background="#d4edda", font=('Consolas', 11, 'bold'))
        self.output_text.tag_configure("heuristic", foreground="#856404", background="#fff3cd", font=('Consolas', 11, 'bold'))
        self.output_text.tag_configure("error", foreground="#721c24", background="#f8d7da", font=('Consolas', 11, 'bold')) 
        self.output_text.tag_configure("result", font=('Consolas', 11))
        self.output_text.tag_configure("distance", foreground="#007BFF", font=('Consolas', 12, 'bold'))
        self.output_text.tag_configure("distance_g", foreground="#FFC107", font=('Consolas', 12, 'bold'))

        self.create_matrix_grid() # Initial call

    def add_new_stop(self):
        """Increments N, captures existing data, and calls create_matrix_grid to redraw."""
        N_old = self.num_stops.get()

        # 1. Capture the current valid matrix data before updating N
        temp_matrix = self.get_matrix_from_entries(N_old)
        if temp_matrix is None: 
             return # Stop if there are input errors
        
        # Store valid data as a list of lists for easy reference
        self.current_matrix_data = temp_matrix.tolist()

        # 2. Increment N and check limits
        N_new = N_old + 1
        if N_new > 8:
            messagebox.showwarning("Limit Reached", "Max number of stops (8) reached for B&B computation speed.")
            return

        # 3. Update the variable and redraw the matrix grid
        self.num_stops.set(N_new)
        self.create_matrix_grid()
        self.update_output(f"New stop '{chr(65+N_old)}' added. Please enter the new distances.", tag="heuristic")


    def create_matrix_grid(self, event=None):
        """Creates the input fields for the distance matrix, pre-filling known values.
        Also updates the start node OptionMenu."""
        try:
            N = self.num_stops.get()
            if N < 3 or N > 8:
                messagebox.showerror("Error", "Please enter N between 3 and 8 for a quick result.")
                self.num_stops.set(4)
                return
        except tk.TclError:
            messagebox.showerror("Error", "Invalid number of stops.")
            return

        # Clear old matrix widgets
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        self.entry_matrix = []
        labels = [chr(65 + i) for i in range(N)] 

        # --- Update Start Node Options (NEW) ---
        menu = self.start_node_menu["menu"]
        menu.delete(0, "end")
        for label in labels:
            menu.add_command(label=label, command=tk._setit(self.start_node_var, label))
        
        # Set default value if the current one is out of bounds
        if self.start_node_var.get() not in labels:
            self.start_node_var.set(labels[0])
        # ---------------------------------------

        # Create Header Row
        tk.Label(self.matrix_frame, text="From \\ To", bg="#e9ecef", font=('Arial', 9, 'bold'), width=5, bd=1, relief=tk.RIDGE).grid(row=0, column=0, padx=1, pady=1)
        for j in range(N):
            tk.Label(self.matrix_frame, text=labels[j], bg="#e9ecef", font=('Arial', 9, 'bold'), width=5, bd=1, relief=tk.RIDGE).grid(row=0, column=j+1, padx=1, pady=1)

        # Create Data Rows
        prev_N = len(self.current_matrix_data) if self.current_matrix_data else 0
        for i in range(N):
            row_entries = []
            tk.Label(self.matrix_frame, text=labels[i], bg="#e9ecef", font=('Arial', 9, 'bold'), width=5, bd=1, relief=tk.RIDGE).grid(row=i+1, column=0, padx=1, pady=1)
            for j in range(N):
                entry = tk.Entry(self.matrix_frame, width=5, justify='center', bd=1, relief=tk.GROOVE)
                entry.grid(row=i+1, column=j+1, padx=1, pady=1)
                
                # Pre-fill existing data
                if i < prev_N and j < prev_N:
                    value = self.current_matrix_data[i][j]
                    entry.insert(0, str(value))
                    # Highlight new cells for clarity, only if we are growing the matrix
                    if N > prev_N and (i == N-1 or j == N-1):
                         entry.config(bg="#FFF8E1") # Light yellow for new inputs
                
                if i == j:
                    entry.insert(0, "0")
                    entry.config(state='readonly', disabledbackground="#f8f9fa")
                    
                row_entries.append(entry)
            self.entry_matrix.append(row_entries)

        self.greedy_button.config(state=tk.NORMAL)
        self.bb_button.config(state=tk.NORMAL)
        if N == prev_N:
             self.update_output("Matrix ready. Select a starting stop and choose a solver option.", tag="result")


    def get_matrix_from_entries(self, N):
        """Extracts the distance matrix from the Entry widgets."""
        matrix = []
        for i in range(N):
            row = []
            for j in range(N):
                try:
                    # Get value, handling readonly state
                    entry_widget = self.entry_matrix[i][j]
                    if i == j: # For diagonal (0) cells
                        val = "0"
                    else:
                        val = entry_widget.get()
                        
                    if not val: raise ValueError
                    row.append(int(val))
                except ValueError:
                    messagebox.showerror("Input Error", f"Distance from {chr(65+i)} to {chr(65+j)} is invalid or empty. Must be an integer.")
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

        # IMPORTANT: After successful extraction, update the stored data
        self.current_matrix_data = D_matrix.tolist() 

        # --- Get the selected start node index (NEW) ---
        start_label = self.start_node_var.get()
        # Convert label ('A'->0, 'B'->1, etc.) to index by finding the difference from 'A'
        start_node_index = ord(start_label) - ord('A') 
        # ---------------------------------------------

        # Determine which solver to run and set up display text
        if method == 'bnb':
            self.update_output(f"Calculating optimal (B&B) starting from {start_label}... Please wait.", tag="result")
            self.master.update() 
            min_distance, path = branch_and_bound_tsp(D_matrix, start_node=start_node_index) # Use the selected index
            header_tag = "success"
            distance_tag = "distance"
            header_text = "🎉 OPTIMAL ROUTE FOUND (Branch & Bound)! 🎉\n"
        elif method == 'greedy':
            self.update_output(f"Calculating heuristic (Greedy) starting from {start_label}... Fast calculation.", tag="result")
            self.master.update() 
            min_distance, path = greedy_tsp(D_matrix, start_node=start_node_index) # Use the selected index
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
                self.output_text.insert(tk.END, "Start Stop: ", "result")
                self.output_text.insert(tk.END, f"{start_label}\n", distance_tag) # Display start node
                self.output_text.insert(tk.END, "Route Distance: ", "result")
                self.output_text.insert(tk.END, f"{min_distance}\n\n", distance_tag)
                
                # Path Result
                self.output_text.insert(tk.END, "Calculated Path:\n", "result")
                self.output_text.insert(tk.END, f"  {' -> '.join(path_labels)}\n", "result")
                self.output_text.insert(tk.END, f"\nPath Length: {N + 1} stops ({path_labels[0]}->...->{path_labels[0]})", "result")

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