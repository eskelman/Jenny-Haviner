
#= Collect data from the CFD file output and collate them into a CSV containing k,ω,λ,y+ =#
ν = 3.50000e-04 # Kinematic viscosity

# From k-ω 
k_data_180 = DataFrames.DataFrame(CSV.File("LM_Channel_0180_vel_fluc_prof.csv")) 
U_data_180 = DataFrames.DataFrame(CSV.File("LM_Channel_0180_mean_prof.csv"))

k_data_550 = DataFrames.DataFrame(CSV.File("LM_Channel_0550_vel_fluc_prof.csv")) 
U_data_550 = DataFrames.DataFrame(CSV.File("LM_Channel_0550_mean_prof.csv"))

k_data_1000 = DataFrames.DataFrame(CSV.File("LM_Channel_1000_vel_fluc_prof.csv")) 
U_data_1000 = DataFrames.DataFrame(CSV.File("LM_Channel_1000_mean_prof.csv"))

k_data_2000 = DataFrames.DataFrame(CSV.File("LM_Channel_2000_vel_fluc_prof.csv")) 
U_data_2000 = DataFrames.DataFrame(CSV.File("LM_Channel_2000_mean_prof.csv"))

k_data_5200 = DataFrames.DataFrame(CSV.File("LM_Channel_5200_vel_fluc_prof.csv")) 
U_data_5200 = DataFrames.DataFrame(CSV.File("LM_Channel_5200_mean_prof.csv"))

import CSV
import DataFrames
    
#instead of having the full tensor variables stuff here, it should intead be a 
function results(k_data,U_data,ν,Re_tau=180) # This input will be edited when the fluid model is ready
    y_plus = k_data."y^+"
    k = k_data."k"
    ω = k./(ν*Re_tau)
    λ = U_data[!,:"dU/dy"]
    CSV.write(joinpath(pwd(),"Code","CFD","results",join(["train_in_Re",Re_tau,".csv"])),DataFrames.DataFrame(y_plus=y_plus,k=k, ω=ω, λ=λ))
    #when a better idea of the structure is known, the naming method will have to be implemented
end 

results(k_data_180,U_data_180,ν,180)
results(k_data_550,U_data_550,ν,550)
results(k_data_1000,U_data_1000,ν,1000)
results(k_data_2000,U_data_2000,ν,2000)
results(k_data_5200,U_data_5200,ν,5200)



