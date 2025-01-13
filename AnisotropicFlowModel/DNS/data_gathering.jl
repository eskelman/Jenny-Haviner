# Collecting the used data from the DNS:

# data - https://turbulence.oden.utexas.edu/

# Myoungkyu Lee and Robert D. Moser  
# "Direct numerical simulation of turbulent channel flow up to Re_tau = 5200"
# 2015 
# Journal of Fluid Mechanics, vol. 774, pp. 395-415 =#

# variables needed:
# u'u', v'v', w'w' - Reynolds stresses to be used in training loss


using Dierckx
using CSV
using DataFrames
root = joinpath(pwd(), "Data", "variance of velocity")
# Variables (could link them um globally)
ν = 3.50000e-04 # Kinematic viscosity

# Create DataFrame from csv
function gather(file, root)
    return CSV.read(joinpath(root, file), DataFrame; header=74)
end

# Create spline dependant on defined component
function spline(df, component)
    return Spline1D(df."y^+", df[:, component]; k=2, bc="nearest", s=0.1)
end

function rs_at_yplus(y_plus,spln)
    return [evaluate(spln,value) for value in y_plus]
end

# Add ω values to the DNS
function add_omega(ν, Re_tau, k)
    return k ./ (ν * Re_tau)
end

# Normalisation function
function normalise(raw_data) # may not be the best normalisation, will require testing
    return (raw_data .- minimum(raw_data)) / (maximum(raw_data) - minimum(raw_data))
end

# Export to CSV
function export_file(name, var_uu, var_vv, var_ww,y_plus)
    CSV.write(joinpath(pwd(),"Code","DNS","results",join([name,".csv"])),DataFrames.DataFrame(var_uu=var_uu, var_vv=var_vv, var_ww=var_ww,y_plus=y_plus))
end

# Load DNS into Re_tau dependant DataFrames and spline the u'u', v'v' and w'w'
# Re_tau = 180
DNS_180 = gather("0180_vel_fluc_prof.csv", root)
DNS_180[!, :ω] .= add_omega(ν, 180, DNS_180.k)
rs_uu_180 = rs_at_yplus(DNS_180."y^+",spline(DNS_180, "u'u'")) #this should be changed to getting y+ from CFD, same goes for export_file(y+)
rs_vv_180 = rs_at_yplus(DNS_180."y^+",spline(DNS_180, "v'v'"))
rs_ww_180 = rs_at_yplus(DNS_180."y^+",spline(DNS_180, "w'w'"))
export_file("train_out_Re180", rs_uu_180,rs_vv_180,rs_ww_180,DNS_180."y^+")

    
# Re_tau = 550
DNS_550 = gather("0550_vel_fluc_prof.csv", root)
DNS_550[!, :ω] .= add_omega(ν, 550, DNS_550.k)
rs_uu_550 = rs_at_yplus(DNS_550."y^+",spline(DNS_550, "u'u'"))
rs_vv_550 = rs_at_yplus(DNS_550."y^+",spline(DNS_550, "v'v'"))
rs_ww_550 = rs_at_yplus(DNS_550."y^+",spline(DNS_550, "w'w'"))
export_file("train_out_Re550", rs_uu_550,rs_vv_550,rs_ww_550,DNS_550."y^+")

# Re_tau = 1000
DNS_1000 = gather("1000_vel_fluc_prof.csv", root)
DNS_1000[!, :ω] .= add_omega(ν, 1000, DNS_1000.k)
rs_uu_1000 = rs_at_yplus(DNS_1000."y^+",spline(DNS_1000, "u'u'"))
rs_vv_1000 = rs_at_yplus(DNS_1000."y^+",spline(DNS_1000, "v'v'"))
rs_ww_1000 = rs_at_yplus(DNS_1000."y^+",spline(DNS_1000, "w'w'"))
export_file("train_out_Re1000", rs_uu_1000,rs_vv_1000,rs_ww_1000,DNS_1000."y^+")

# Re_tau = 2000
DNS_2000 = gather("2000_vel_fluc_prof.csv", root)
DNS_2000[!, :ω] .= add_omega(ν, 2000, DNS_2000.k)
rs_uu_2000 = rs_at_yplus(DNS_2000."y^+",spline(DNS_2000, "u'u'"))
rs_vv_2000 = rs_at_yplus(DNS_2000."y^+",spline(DNS_2000, "v'v'"))
rs_ww_2000 = rs_at_yplus(DNS_2000."y^+",spline(DNS_2000, "w'w'"))
export_file("train_out_Re2000", rs_uu_2000,rs_vv_2000,rs_ww_2000,DNS_2000."y^+")

# Re_tau = 5200
DNS_5200 = gather("5200_vel_fluc_prof.csv", root)
DNS_5200[!, :ω] .= add_omega(ν, 5200, DNS_5200.k)
rs_uu_5200 = rs_at_yplus(DNS_5200."y^+",spline(DNS_5200, "u'u'"))
rs_vv_5200 = rs_at_yplus(DNS_5200."y^+",spline(DNS_5200, "v'v'"))
rs_ww_5200 = rs_at_yplus(DNS_5200."y^+",spline(DNS_5200, "w'w'"))
export_file("train_out_Re5200", rs_uu_5200,rs_vv_5200,rs_ww_5200,DNS_5200."y^+")

# Can evaluate the funciton at any point
evaluate(rs_vv_5200, [500,0.5])

DNS_5200