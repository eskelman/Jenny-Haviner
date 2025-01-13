
using DataFrames, CSV, Random, Flux, DataFramesMeta
using .RS
using Plots
using Statistics
using StatsBase

# Functions
function gather(file, root)
    return DataFrame(CSV.File(joinpath(root, file)))
end

cd(joinpath("/home", "esk","Documents", "Individual Project", "RS-Model-1"))

function mix(folder1,folder2) # mixes the files in one folder into one DataFrame'
    dir1 = readdir(folder1)
    dir2 = readdir(folder2)
    master=DataFrame()
    for i in eachindex(dir1)
        CFD = CSV.read(joinpath(folder1,dir1[i]),DataFrame)
        DNS = CSV.read(joinpath(folder2,dir2[i]),DataFrame)
        combo = hcat(CFD,DNS[!, Not("y_plus")]) 
        append!(master, combo)
    end
    return master[shuffle(axes(master, 1)), :]
end

function export_results(y_plus, prediction, truth, name="RS_results_normalised")
    CSV.write(
        joinpath(pwd(),"Code", "Post Processing", "results",join([name,".csv"])),
        DataFrames.DataFrame(y_plus=y_plus, uu_yhat=prediction[1,:], vv_yhat=prediction[2,:], ww_yhat=prediction[3,:], uu_y=truth[:,1], vv_y=truth[:,2], ww_y=truth[:,3]))
end

base = joinpath(pwd(),"Code");

# Collate and shuffle the training data
training_data = mix(joinpath(base,"CFD","results"),joinpath(base,"DNS","results"))

# Normalisation
function norm(input, target_max, target_min)
    input_max = maximum(input, dims=1)
    input_min = minimum(input, dims=1)
    #println("min: $input_min, max: $input_max")
    normalised_array = ((input.-input_min)./(input_max.-input_min))*(target_max - target_min)
    return normalised_array,input_max,input_min
end

function unnorm(input, orig_max, orig_min)
    return input.*(orig_max .- orig_min) .+ orig_min
end

# Training Input/Output
training_output = hcat(training_data.var_uu, training_data.var_vv, training_data.var_ww)
k = training_data.k
ω = training_data.ω
yplus = training_data.y_plus
λ = training_data.λ
training_input = hcat(yplus, k , ω)
training_input_norm = hcat(norm(yplus,1,0)[1],norm(k,1,0)[1],norm(ω,1,0)[1])

# Isolates Re180 data for comparison with Dhila Model
Re180_training_data = gather("train_in_Re180.csv",joinpath(base, "CFD", "results"))
yplus_Re180 = Re180_training_data.y_plus
k_Re180 = Re180_training_data.k
ω_Re180 = Re180_training_data.ω
λ_Re180 = Re180_training_data.λ

#Testing the normalisation function

k_norm, k_max, k_min = norm(k,1,0)
ω_norm, ω_max, ω_min = norm(ω,1,0)
yplus_norm, yplus_max, yplus_min = norm(yplus,1,0)

g = k .- unnorm(k_norm, k_max, k_min)
h = ω .- unnorm(ω_norm, ω_max, ω_min)
i = yplus .- unnorm(yplus_norm, yplus_max, yplus_min)
maximum(g)


function train(opt,act,epochs,batchsize,hidden_nodes,loss_function, yplus, k, ω, training_output, λ, yplus_test, k_test, ω_test, λ_test)
    # Create the data batch to be processed by the network
    loaded_data = Flux.DataLoader((yplus, k, ω, λ, training_output'), batchsize=batchsize)
    
    # Model Architecture
    nINPUTS = 3;
    nHIDDEN = hidden_nodes;
    nOUTPUTS = 2;

    model = Flux.Chain(Dense(nINPUTS, nHIDDEN, act),
        Dense(nHIDDEN, nHIDDEN, act),
        Dense(nHIDDEN, nHIDDEN, act),
        Dense(nHIDDEN, nHIDDEN, act),
        Dense(nHIDDEN, nOUTPUTS, act))

    # Loss Function
    loss(yplus, k, ω, λ, validation) = loss_function(RS.calculation(
        hcat(yplus, k , ω)', # non normalised input
        model(hcat(norm(yplus,1,0)[1],norm(k,1,0)[1],norm(ω,1,0)[1])'),λ),# normalised input
        validation # validation target
        )

    # Optimiser and parameters
    prms = Flux.params(model) # retrieve the network params of the previously defined model

    # Training loop
    loss_history = [];

    time = @elapsed begin
    for epoch in 1:epochs
        #train model
        Flux.train!(loss, prms, loaded_data, opt)

        # Initiate loss function and store the residuals 
        train_loss = loss(yplus, k, ω, λ, training_output')
        push!(loss_history, train_loss)
        println("Epoch = $epoch : Training Loss = $train_loss")
    end
    end
    # Return historic losses over the iterations 
    return loss_history,time,
        RS.calculation(
            hcat(yplus, k , ω)',
            model(hcat(norm(yplus,1,0)[1],norm(k,1,0)[1],norm(ω,1,0)[1])'),
            λ),

        RS.calculation(
            hcat(yplus_test, k_test, ω_test)',
            model(hcat(norm(yplus_test,1,0)[1],norm(k_test,1,0)[1],norm(ω_test,1,0)[1])'),
            λ_test)
end

# Normal Run
optimiser = RMSProp()
activation_function = leakyrelu
loss_funciton = Flux.mse

epochs = 1000
batchsize = 300
hidden_nodes = 200

l_norm,t_norm,p_norm,p180_norm = train(optimiser, activation_function, epochs, batchsize, hidden_nodes, loss_funciton, yplus, k, ω, training_output, λ, yplus_Re180, k_Re180, ω_Re180, λ_Re180)
loss_plot = plot(1:length(l_norm), l_norm, label = "Normalised Model Loss ($t_norm s, pmax = $(maximum(p_norm)))")
maximum(p_norm)
export_results(yplus,p_norm,training_output)
export_results(yplus_Re180,p180_norm,Re180_training_input, "Re180_norm")

# Optimiser analysis
opt_options = Dict("ADAM" => ADAM(),"Descent" => Descent(), "Momentum" => Momentum(), "AdaMax" => AdaMax(), "RMSProp" => RMSProp())
redirectopt_analysis = plot()

for key in keys(opt_options)
    println(key)
    l,t,p= train(opt_options[key], leakyrelu, 1000, 200,100,Flux.mse,training_input, training_output,λ)
    plot!(1:length(l), l, label = "$key ($t s, pmax = $(maximum(p)))")
    display(current())
end

# Activation function analysis
act_options = Dict("relu" => relu,"sigmoid" => sigmoid, "tanh" => tanh, "softplus" => softplus, "leakyrelu" => leakyrelu)
act_analysis = plot()

for key in keys(act_options)
    l,t,p = train(ADAM(), act_options[key], 1000, 200, 100,Flux.mse,training_input,training_output,λ)
    plot!(1:length(l), l, label = "$key ($t s, pmax = $(maximum(p)))")
    display(current())
end

# Loss function analysis
loss_options = Dict("mse" => Flux.mse,"mae" => Flux.mae, "huber" => Flux.huber_loss)
loss_analysis = plot()

for key in keys(loss_options)
    l,t,p = train(ADAM(), leakyrelu, 1000, 200, 100,loss_options[key],training_input,training_output,λ)
    plot!(1:length(l), l, label = "$key ($t s, pmax = $(maximum(p)))")
    display(current())
end