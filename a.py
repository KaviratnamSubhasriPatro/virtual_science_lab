import streamlit as st
import base64
import numpy as np
import matplotlib.pyplot as plt
import cmath
import heapq
import math
# Function to encode image into Base64
def get_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load background images
bg1_path = "bg.jpg"   # Background for Home Page
bg2_path = "b1g.jpg"  # Background for Experiments Page

bg1_base64 = get_base64(bg1_path)
bg2_base64 = get_base64(bg2_path)

# Initialize session state if not set
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Function to navigate to experiments
def go_to_experiments():
    st.session_state.page = "Experiments"

# Apply different background images based on the selected page
if st.session_state.page == "Home":
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpeg;base64,{bg1_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Welcome to Virtual Science Lab")
    st.write("Explore the fascinating world of Physics, Chemistry, and Math through interactive experiments.")
    
    if st.button("Start Experiments 🚀"):
        go_to_experiments()

elif st.session_state.page == "Experiments":
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/jpeg;base64,{bg2_base64}") no-repeat center center fixed;
        background-size: cover;
        background-size:100% 100%;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("🧪 Virtual Science Lab")
    st.write("Perform various science experiments interactively.")
    
    # Experiment Categories
    category = st.radio("Select a Category", ["Physics", "Chemistry", "Math", "CSE"], horizontal=True)
# Define Gravity
GRAVITY = 9.81

# ---------------- Physics Experiments ----------------
if 'category' in locals() and category == "Physics":
    experiment = st.selectbox("Select an Experiment", [
        "Free Fall", "Ohm's Law", "Simple Pendulum", "Projectile Motion", "Newton's Second Law"
    ])
    
    if experiment == "Free Fall":
        st.header("📌 Physics: Free Fall Experiment")
        height = st.number_input("Enter height (meters):", min_value=0.1, value=10.0)
        time_of_fall = np.sqrt(2 * height / GRAVITY)
        st.write(f"🕒 Time to reach ground: {time_of_fall:.2f} seconds")

    elif experiment == "Ohm's Law":
        st.header("📌 Physics: Ohm's Law Experiment")
        choice = st.radio("Select what to calculate:", ["Voltage (V)"])
        if choice == "Voltage (V)":
            I = st.number_input("Enter Current (A):", min_value=0.1, value=1.0)
            R = st.number_input("Enter Resistance (Ω):", min_value=0.1, value=10.0)
            st.write(f"🔋 Voltage (V) = {I * R:.2f} Volts")

    elif experiment == "Simple Pendulum":
        st.header("📌 Physics: Simple Pendulum Experiment")
        length = st.number_input("Enter length of pendulum (meters):", min_value=0.1, value=1.0)
        time_period = 2 * np.pi * np.sqrt(length / GRAVITY)
        st.write(f"🕰 Time Period: {time_period:.2f} seconds")

        # Graph
        angles = np.linspace(-np.pi/4, np.pi/4, num=50)
        displacement = length * np.sin(angles)

        fig, ax = plt.subplots()
        ax.plot(angles, displacement, marker="o", color="g")
        ax.set_xlabel("Angle (radians)")
        ax.set_ylabel("Displacement (m)")
        ax.set_title("Simple Pendulum Motion")
        ax.grid()
        st.pyplot(fig)
    
    elif experiment == "Newton's Second Law":
        st.header("📌 Physics: Newton's Second Law")
        force = st.number_input("Enter Force (N):", min_value=0.1, value=10.0)
        mass = st.number_input("Enter Mass (kg):", min_value=0.1, value=1.0)
        acceleration = force / mass
        st.write(f"🚀 Acceleration: {acceleration:.2f} m/s²")

    elif experiment == "Projectile Motion":
        st.header("📌 Physics: Projectile Motion")
        angle = st.number_input("Enter launch angle (degrees):", min_value=1, max_value=89, value=45)
        velocity = st.number_input("Enter initial velocity (m/s):", min_value=0.1, value=10.0)
        theta = np.radians(angle)
        time_of_flight = (2 * velocity * np.sin(theta)) / GRAVITY
        max_height = (velocity**2 * np.sin(theta)**2) / (2 * GRAVITY)
        range_ = (velocity**2 * np.sin(2 * theta)) / GRAVITY
        
        st.write(f"🕒 Time of flight: {time_of_flight:.2f} seconds")
        st.write(f"📏 Maximum height: {max_height:.2f} meters")
        st.write(f"🏹 Range: {range_:.2f} meters")

# ---------------- Chemistry Experiments ----------------
elif 'category' in locals() and category == "Chemistry":
    experiment = st.selectbox("Select an Experiment", ["pH Calculation", "Molarity Calculation"])
    
    if experiment == "pH Calculation":
        st.header("📌 Chemistry: pH Calculation")
        H_concentration = st.number_input("Enter Hydrogen Ion Concentration (mol/L):", min_value=1e-14, value=1e-7, format="%.2e")
        pH = -np.log10(H_concentration)
        st.write(f"🧪 pH of the solution: {pH:.2f}")
        
        if pH < 7:
            st.write("🔴 The solution is Acidic.")
        elif pH == 7:
            st.write("🟢 The solution is Neutral.")
        else:
            st.write("🔵 The solution is Basic.")
    
    elif experiment == "Molarity Calculation":
        st.header("📌 Chemistry: Molarity Calculation")
        moles = st.number_input("Enter number of moles of solute:", min_value=0.01, value=1.0)
        volume = st.number_input("Enter volume of solution (liters):", min_value=0.01, value=1.0)
        molarity = moles / volume
        st.write(f"⚗ Molarity = {molarity:.2f} M")

# ---------------- Math Experiments ----------------
elif 'category' in locals() and category == "Math":
    experiment = st.selectbox("Select an Experiment", ["Quadratic Equation Solver", "Probability Calculation","Matrix Multiplication","Factorial of A Number"])
    
    if experiment == "Quadratic Equation Solver":
        st.header("📌 Math: Quadratic Equation Solver")
        a = st.number_input("Enter coefficient a:", value=1.0, format="%.2f")
        b = st.number_input("Enter coefficient b:", value=0.0, format="%.2f")
        c = st.number_input("Enter coefficient c:", value=0.0, format="%.2f")
        discriminant = (b**2) - (4*a*c)
        sqrt_discriminant = cmath.sqrt(discriminant)
        root1 = (-b + sqrt_discriminant) / (2*a)
        root2 = (-b - sqrt_discriminant) / (2*a)
        st.write(f"🌿 Root 1: {root1.real:.2f}, Root 2: {root2.real:.2f}")
    
    elif experiment == "Probability Calculation":
        st.header("📌 Math: Probability Calculation")
        favorable = st.number_input("Enter number of favorable outcomes:", min_value=0, value=1)
        total = st.number_input("Enter total number of outcomes:", min_value=1, value=2)
        probability = favorable / total
        st.write(f"🎲 Probability: {probability:.2f}")

    elif experiment == "Matrix Multiplication":
        st.header("📌 Math: Matrix Multiplication")
        rows_A = st.number_input("Enter number of rows for Matrix A:", min_value=1, value=2)
        cols_A = st.number_input("Enter number of columns for Matrix A:", min_value=1, value=2)
        rows_B = st.number_input("Enter number of rows for Matrix B:", min_value=1, value=2)
        cols_B = st.number_input("Enter number of columns for Matrix B:", min_value=1, value=2)
        
        if cols_A != rows_B:
            st.error("Matrix multiplication not possible! Columns of A must match rows of B.")
        else:
            st.write("Enter elements for Matrix A:")
            matrix_A = np.array([[st.number_input(f"A[{i}][{j}]", value=1) for j in range(cols_A)] for i in range(rows_A)])
            
            st.write("Enter elements for Matrix B:")
            matrix_B = np.array([[st.number_input(f"B[{i}][{j}]", value=1) for j in range(cols_B)] for i in range(rows_B)])
            
            result = np.dot(matrix_A, matrix_B)
            st.write("Matrix A:", matrix_A)
            st.write("Matrix B:", matrix_B)
            st.write("Product:", result)
    
    elif experiment == "Factorial of A Number":
        st.header("📌 Math: Factorial Calculation")
        num = st.number_input("Enter a number:", min_value=0, value=5)
        factorial = math.factorial(num)
        st.write(f"💡 Factorial of {num}: {factorial}")    


if 'category' in locals() and category == "CSE":
    experiment = st.selectbox("Select an Experiment", ["Dijkstra's Algorithm", "Binary Search", "Bubble Sort"])
    
    if experiment == "Dijkstra's Algorithm":
        st.header("💻 CSE: Dijkstra's Shortest Path Algorithm")
        st.write("Find the shortest path in a weighted graph using Dijkstra's Algorithm.")
        
        num_nodes = st.number_input("Enter number of nodes:", min_value=2, value=5, step=1)
        edges = st.text_area("Enter edges (format: u v w, one per line):", "0 1 2\n0 2 5\n1 2 1\n1 3 3\n2 3 2")
        start_node = st.number_input("Enter starting node:", min_value=0, value=0, step=1)
        
        if st.button("Run Dijkstra"):
            graph = {i: [] for i in range(num_nodes)}
            for line in edges.split("\n"):
                try:
                    u, v, w = map(int, line.split())
                    graph[u].append((v, w))
                    graph[v].append((u, w))  # Assuming undirected graph
                except:
                    st.error("Invalid edge format. Use u v w")
                    break
            
            def dijkstra(graph, start):
                distances = {node: float('inf') for node in graph}
                distances[start] = 0
                pq = [(0, start)]
                while pq:
                    curr_dist, node = heapq.heappop(pq)
                    if curr_dist > distances[node]:
                        continue
                    for neighbor, weight in graph[node]:
                        distance = curr_dist + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            heapq.heappush(pq, (distance, neighbor))
                return distances
            
            result = dijkstra(graph, start_node)
            st.write("Shortest distances from node", start_node, ":", result)

    elif experiment == "Binary Search":
        st.header("💻 CSE: Binary Search")
        st.write("Find an element in a sorted array using Binary Search.")
        
        array = st.text_input("Enter sorted array (comma-separated):", "1, 3, 5, 7, 9, 11, 13")
        target = st.number_input("Enter target value:", value=5)
        
        if st.button("Search"):
            try:
                arr = sorted(map(int, array.split(',')))
                
                def binary_search(arr, target):
                    left, right = 0, len(arr) - 1
                    while left <= right:
                        mid = (left + right) // 2
                        if arr[mid] == target:
                            return mid
                        elif arr[mid] < target:
                            left = mid + 1
                        else:
                            right = mid - 1
                    return -1
                
                index = binary_search(arr, target)
                if index != -1:
                    st.success(f"Element found at position {index+1}!")
                else:
                    st.error("Element not found.")
            except:
                st.error("Invalid input. Please enter numbers separated by commas.")
    
    elif experiment == "Bubble Sort":
        st.header("💻 CSE: Bubble Sort")
        st.write("Sort an array using the Bubble Sort algorithm.")
        
        array = st.text_input("Enter array (comma-separated):", "5, 3, 8, 1, 2")
        
        if st.button("Sort"):
            try:
                arr = list(map(int, array.split(',')))
                
                def bubble_sort(arr):
                    n = len(arr)
                    for i in range(n):
                        for j in range(0, n-i-1):
                            if arr[j] > arr[j+1]:
                                arr[j], arr[j+1] = arr[j+1], arr[j]
                    return arr
                
                sorted_arr = bubble_sort(arr)
                st.success(f"Sorted Array: {sorted_arr}")
            except:
                st.error("Invalid input. Please enter numbers separated by commas.")
