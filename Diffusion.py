import fenics as fe
import dolfin as df
from fenics import MixedElement, FunctionSpace, TestFunctions, Function, split
from fenics import derivative, NonlinearVariationalProblem, NonlinearVariationalSolver
from fenics import UserExpression, sqrt
import numpy as np
from tqdm import tqdm


#################### Define Domain ####################
dy = 0.05 

Nx_aprox = 10 
Ny_aprox = 10 

nx = (int)(Nx_aprox / dy ) + 1
ny = (int)(Ny_aprox / dy ) + 1

Nx = nx * dy

Ny = ny * dy

mesh = fe.RectangleMesh( df.Point(0, 0), df.Point(Nx, Ny), nx, ny )


#############################  END  #############################
 


#################### Define Constants ####################

D = 4  # Thermal diffusivity of steel, mm2.s-1

T_hot = 700 

T_cold = 300 

dt = 1/(4*D) * dy**2

rad_init = 2

center = [ 5.0, 5.0 ]  # Circle center

#############################  END  #############################


#################### Define Variables  ####################


def variable_define(mesh):
    
    """
    Define variables and function spaces for a given mesh.

    Parameters:
    mesh: The computational mesh

    Returns:
    Tuple containing various elements, function spaces, and test functions.
    """

    # Define finite element
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # U or the Concentration

    # Create a mixed finite element
    element = MixedElement([P1, P1])

    # Define function space
    ME = FunctionSpace(mesh, element)

    # Define test function
    v_test , w_test= TestFunctions(ME)

    # Define current and previous solutions
    Sol_Func = Function(ME)  # Current solution
    Sol_Func_0 = Function(ME)  # Previous Solution

    # Split functions to get individual components
    U_answer , Phi = split(Sol_Func)  # current solution
    U_prev , Phi_0 = split(Sol_Func_0)  # last step solution



    # Extract subspaces and mappings for each subspace
    num_subs = ME.num_sub_spaces()
    spaces, maps = [], []
    for i in range(num_subs):
        space_i, map_i = ME.sub(i).collapse(collapsed_dofs=True)
        spaces.append(space_i)
        maps.append(map_i)

    # Return all the defined variables
    return U_answer, U_prev, Phi, Phi_0,  Sol_Func, Sol_Func_0, v_test, w_test, spaces, ME





def eq_diff(u_answer, u_prev, v_test, dt, D):

    """
    Calculate the differential equation term.

    Parameters:
    u_answer: The current solution
    u_prev: The solution from the previous time step
    v_test: Test function
    dt: Time step size
    D: Diffusion coefficient

    Returns:
    The finite element form of the differential equation.
    """

    # Calculate the time derivative term
    time_derivative = (u_answer - u_prev) / dt

    # Calculate the spatial derivative term
    spatial_derivative = fe.grad(u_answer)

    # Assemble the equation
    eq = ( fe.inner(time_derivative, v_test ) + 
          
          D * fe.inner(spatial_derivative, fe.grad(v_test ) ) ) * fe.dx
    

    return eq


def eq_Phi(u_answer, u_prev, v_test, dt, D):


    # Calculate the time derivative term
    time_derivative = (u_answer - u_prev) / dt


    eq = fe.inner(time_derivative, v_test ) * fe.dx
          
 
    

    return eq
#############################  END  #############################

#################### Define Problem and Solver  ####################


def problem_define(eq1, eq2,  sol_func):

    """
    Define a nonlinear variational problem and its solver.

    Parameters:
    eq1: The left-hand side of the equation (weak form)
    sol_func: The solution function

    Returns:
    A configured nonlinear variational solver.
    """

    # Define the variational problem
    L = eq1 + eq2
    J = derivative(L, sol_func)  # Compute the Jacobian of L
    problem = NonlinearVariationalProblem(L, sol_func, J=J)

    # Configure the solver for the problem
    solver = NonlinearVariationalSolver(problem)

    # Access and set solver parameters
    prm = solver.parameters
    prm["newton_solver"]["relative_tolerance"] = 1e-5
    prm["newton_solver"]["absolute_tolerance"] = 1e-6
    prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True

    return solver


#############################  END  #############################


#################### Define Initial Condition  ####################


class InitialConditions(UserExpression):
    def __init__(self, rad, center, T_hot, T_cold, **kwargs):
        super().__init__(**kwargs)
        self.rad = rad  # Initial circle radius
        self.center = center  # Center of the circle (xc, yc)
        self.T_hot = T_hot  # Temperature inside the circle
        self.T_cold = T_cold  # Temperature outside the circle

    def eval(self, values, x):
        xc, yc = self.center
        x, y = x[0], x[1]  # Coordinates
        dist = (x - xc)**2 + (y - yc)**2  # Distance squared from the center

        # Check if the point is inside the circle (dist <= rad^2)
        if dist <= self.rad**2:
            values[0] = self.T_hot  # Inside the circle
        else:
            values[0] = self.T_cold  # Outside the circle

        values[1] = 0  # If 'Phi' is the second variable and it's always 0

    def value_shape(self):
        return (2,)

    
def Initial_Interpolate( sol_func, Sol_Func_0, rad, center, T_hot, T_cold,  degree ):


    initial_conditions = InitialConditions(rad= rad, center = center, T_hot= T_hot,T_cold= T_cold ,  degree= degree)

    sol_func.interpolate(initial_conditions)

    Sol_Func_0.interpolate(initial_conditions)



#############################  END  ###############################


#################### Define Step 1 For Solving  ####################
    
U_answer, U_prev, Phi, Phi_0,  Sol_Func, Sol_Func_0, v_test, w_test, spaces, ME = variable_define(mesh= mesh )

eqdiff = eq_diff(u_answer= U_answer, u_prev= U_prev, v_test= v_test, dt= dt , D= D )

eqPhi = eq_Phi(u_answer= Phi , u_prev= Phi_0, v_test= w_test , dt = dt, D= D)

Initial_Interpolate(Sol_Func, Sol_Func_0 , rad_init, center, T_hot, T_cold, 2 )

solver = problem_define(eq1= eqdiff, eq2= eqPhi, sol_func= Sol_Func)

#############################  END  ###############################

############################ File Section #########################

def write_simulation_data(Sol_Func, time, file_path, variable_names ):
    """
    Writes the simulation data to an XDMF file. Handles an arbitrary number of variables.

    Parameters:
    - Sol_Func : fenics.Function
        The combined function of variables (e.g., Phi, U, Theta).
    - time : float
        The simulation time or step to associate with the data.
    - file_path : str, optional
        The path to the XDMF file where data will be written.
    - variable_names : list of str, optional
        The names of the variables in the order they are combined in Sol_Func.
    """

    # Open the file for writing simulation data
    with fe.XDMFFile(file_path) as file:
        # Configure file parameters
        file.parameters["rewrite_function_mesh"] = True
        file.parameters["flush_output"] = True
        file.parameters["functions_share_mesh"] = True

        # Split the combined function into its components
        functions = Sol_Func.split(deepcopy=True)

        # Check if the number of variable names matches the number of functions
        if variable_names and len(variable_names) != len(functions):
            raise ValueError("The number of variable names must match the number of functions.")

        # Rename and write each function to the file
        for i, func in enumerate(functions):
            name = variable_names[i] if variable_names else f"Variable_{i}"
            func.rename(name, "solution")
            file.write(func, time)


T = 0

variable_names = [ "U", "Phi" ]  # Adjust as needed

file_path= "Diffusion.xdmf" 

write_simulation_data( Sol_Func, T, file_path , variable_names=variable_names )


#############################  END  ###############################



############################ Solever Loop #########################

for it in tqdm(range(0, 1000000000)):

    T = T + dt

    solver.solve()

    Sol_Func_0.vector()[:] = Sol_Func.vector()  # update the solution


    if it % 20 == 0 :

        write_simulation_data( Sol_Func_0,  T , file_path , variable_names )



#############################  END  ###############################

