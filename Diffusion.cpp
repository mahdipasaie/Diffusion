#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>  
#include <filesystem> 


const double dx = 0.05 ;
const double dy = 0.05 ;
const double D= 4 ; 
const double T_inside = 700 ;
const double T_outside = 300 ;
const double dt = 1/(4*D) * std::pow(dx, 2) ;
const double Lx = 10.0;   
const double Ly = 10.0;   
const int Nx = static_cast<int>(std::ceil(Lx / dx)) + 1;
const int Ny = static_cast<int>(std::ceil(Ly / dy)) + 1;
const double radius = 2; 
const double flux = 0.0; 
const double T = 0.1;    


//  function to update the bulk field of diffusion 
std::vector<std::vector<double>> rhs( std::vector<std::vector<double>> T, double dx, 
    double dy, double alpha, double dt, int Nx, int Ny );
// function for initial condition
std::vector<std::vector<double>> initialcondition(std::vector<std::vector<double>> T, int Nx, int Ny, double dx, double dy, 
    double radius, double T_inside, double T_outside);
//function to write to csv file
void write_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data);
// Boundary condition
std::vector<std::vector<double>> apply_neumann_bc(std::vector<std::vector<double>> T, double flux, std::string side, double dx, double dy );





int main(){
    
    std::filesystem::create_directory("output"); 
    std::cout << "The time step is : " << dt << "  " << std::endl ;
    std::vector<std::vector<double>> Temperature(Ny, std::vector<double>(Nx, 0));
    std::vector<std::vector<double>> T_(Ny, std::vector<double>(Nx, 0));
    Temperature = initialcondition(Temperature, Nx, Ny, dx, dy, radius, T_inside, T_outside);
    write_to_csv("output/temperature.csv", Temperature);
    // Boundary condition 
    Temperature = apply_neumann_bc(Temperature, flux, "left", dx, dy ) ;
    Temperature = apply_neumann_bc(Temperature, flux, "right", dx, dy ) ;
    Temperature = apply_neumann_bc(Temperature, flux, "bottom", dx, dy ) ;
    Temperature = apply_neumann_bc(Temperature, flux, "top", dx, dy ) ;
    write_to_csv("output/temperature.csv", Temperature);

    // Time Loop:
    double Time = 0 ;
    int it = 0 ;

    while (Time < T)
    {
        it += 1 ;
        Time += dt ;

        T_ = rhs(Temperature, dx, dy, D, dt, Nx, Ny ) ; // next time step

        // Boundary condition 
        Temperature = apply_neumann_bc(Temperature, flux, "left", dx, dy ) ;
        Temperature = apply_neumann_bc(Temperature, flux, "right", dx, dy ) ;
        Temperature = apply_neumann_bc(Temperature, flux, "bottom", dx, dy ) ;
        Temperature = apply_neumann_bc(Temperature, flux, "top", dx, dy ) ;
        // update for the next time step
        Temperature = T_;
        // save the time step: 
        std::ostringstream filename;
        filename << "output/temperature_" << std::setfill('0') << std::setw(4) << it << ".csv";
        write_to_csv(filename.str(), Temperature);


    }
    
    return 0;

}


// FUNCTIONS :

std::vector<std::vector<double>> rhs( std::vector<std::vector<double>> T, double dx, double dy, double alpha, double dt, int Nx, int Ny ){
    
    double coeff_x = alpha* dt / std::pow(dx, 2) ; 
    double coeff_y = alpha* dt / std::pow(dy, 2) ; 
    std::vector<std::vector<double>> T_(Ny, std::vector<double>(Nx, 0));

    for (int j = 1; j < Ny - 1; j++) { 
        for (int i = 1; i < Nx - 1; i++) { 
            T_[j][i] = 
                coeff_x * (T[j][i + 1] - 2 * T[j][i] + T[j][i - 1]) +
                coeff_y * (T[j + 1][i] - 2 * T[j][i] + T[j - 1][i]) +
                T[j][i];
        }
    }

    return T_ ;

}

std::vector<std::vector<double>> initialcondition(
    std::vector<std::vector<double>> T, int Nx, int Ny, double dx, double dy, 
    double radius, double T_inside, double T_outside) 
{
    double cx = Nx * dx / 2.0;
    double cy = Ny * dy / 2.0;

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            double x = i * dx;
            double y = j * dy;

            double distance_squared = std::pow(cx - x, 2) + std::pow(cy - y, 2);

            if (distance_squared <= std::pow(radius, 2)) {
                T[j][i] = T_inside;  // Inside the circle
            } else {
                T[j][i] = T_outside; // Outside the circle
            }
        }
    }

    return T;
}



void write_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            fout << row[i];
            if (i < row.size() - 1)
                fout << ",";
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "Data written to " << filename << std::endl;
}

std::vector<std::vector<double>> apply_neumann_bc(std::vector<std::vector<double>> T, double flux, std::string side, double dx, double dy ){

    if( side == "left" ){
        for (int i=0; i<Ny ; i++ ){ // Neumann BC at x = 0

            T[i][0] = T[i][1] + flux* dx ;
        }
    }
    else if (side == "right") {  // Neumann BC at x = Lx
        for (int i = 0; i < Ny; i++) {
            T[i][Nx - 1] = T[i][Nx - 2] + flux * dx;
        }
    }
    else if (side == "bottom") {  // Neumann BC at y = 0
        for (int j = 0; j < Nx; j++) {
            T[0][j] = T[1][j] + flux * dy;
        }
    }
    else if (side == "top") {  // Neumann BC at y = Ly
        for (int j = 0; j < Nx; j++) {
            T[Ny - 1][j] = T[Ny - 2][j] + flux * dy;
        }
    }
    else {
        std::cerr << "Error: Invalid boundary side. Use 'left', 'right', 'top', or 'bottom'.\n";
    }

    return T;

}




