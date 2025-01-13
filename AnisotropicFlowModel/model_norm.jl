# Train a model based on the data gathering from the DNS and CFD
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

# Training Input

maximum(output[:,1])

k = training_data.k
ω = training_data.ω
yplus = training_data.y_plus
λ = training_data.λ

# Training Comparison (Output)
uu = training_data.var_uu
vv = training_data.var_vv
ww = training_data.var_ww
output = hcat(uu, vv, ww)

# Isolates Re180 data for comparison with Dhila Model
Re180_training_data_in = gather("train_in_Re180.csv",joinpath(base, "CFD", "results"))
Re180_training_data_out = gather("train_out_Re180.csv",joinpath(base, "DNS", "results"))
yplus_Re180 = Re180_training_data_in.y_plus
k_Re180 = Re180_training_data_in.k
ω_Re180 = Re180_training_data_in.ω
λ_Re180 = Re180_training_data_in.λ
output_Re180 = hcat(Re180_training_data_out.var_uu, Re180_training_data_out.var_vv, Re180_training_data_out.var_ww)

#Testing the normalisation function

#Expected Range
emax = 1
emin = 0

inputs = hcat(yplus,k,ω)
inputs_norm = hcat(norm(yplus,emax,emin)[1],norm(k,emax,emin)[1],norm(ω,emax,emin)[1])
outputs_norm = hcat(norm(uu,emax,emin)[1],norm(vv,emax,emin)[1],norm(ww,emax,emin)[1])
inputs_test = hcat(yplus_Re180,k_Re180,ω_Re180)
inputs_test_norm = hcat(norm(yplus_Re180,emax,emin)[1],norm(k_Re180,emax,emin)[1],norm(ω_Re180,emax,emin)[1])

function train(opt,act,epochs,batchsize,hidden_nodes,loss_function, inputs, inputs_norm, output, λ, inputs_test, inputs_test_norm, λ_test)
    # Create the data batch to be processed by the network
    loaded_data = Flux.DataLoader((inputs', inputs_norm', λ, output'), batchsize=batchsize)
    
    # Model Architecture
    nINPUTS = 3;
    nHIDDEN = hidden_nodes;
    nOUTPUTS = 2;

    model = Flux.Chain(Dense(nINPUTS, 128, act),
        Dense(128, 64, act),
        Dense(64, 64, act),
        Dense(64, 64, act),
        Dense(64, nOUTPUTS,act)
        )

    # Loss Function
    loss(inputs, inputs_norm, λ, validation) = loss_function(RS.calculation(
        inputs, # non normalised input
        model(hcat(inputs_norm)),# normalised input
        λ),
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
        train_loss = loss(inputs', inputs_norm', λ, output')
        push!(loss_history, train_loss)
        println("Epoch = $epoch : Training Loss = $train_loss")
    end
    end
    # Return historic losses over the iterations 
    return loss_history,time,
        RS.calculation(
            inputs',
            model(inputs_norm'),
            λ),

        RS.calculation(
            inputs_test',
            model(inputs_test_norm'),
            λ_test)
end

# Normal Run
optimiser = Descent(0.01)
activation_function = leakyrelu
loss_funciton = Flux.mse

epochs = 5000
batchsize = 10
hidden_nodes = 300

l_n,t_n,p_n,p180_n = train(optimiser, activation_function, epochs, batchsize, hidden_nodes, loss_funciton, inputs, inputs_norm, output, λ, inputs_test, inputs_test_norm, λ_Re180)
loss_plot = plot(1:length(l_n), l_n, label = "Normalised Model Loss ($t_n s, pmax = $(maximum(p_n)))")
maximum(abs.(p_n[1,:]))
export_results(yplus,p_n,output')
p180_n
maximum(abs.(p180_n[1,:]))
export_results(yplus_Re180,p180_n,output_Re180', "Re180_norm")

# Optimiser analysis
opt_options = Dict("ADAM" => ADAM(0.01),#="Descent" => Descent(), "Momentum" => Momentum(),=# "AdaMax" => AdaMax(0.01), "RMSProp" => RMSProp(0.01))
redirectopt_analysis = plot()

for key in keys(opt_options)
    println(key)
    l,t,p= train(opt_options[key], leakyrelu, 1000, 300, 100,Flux.mse, inputs, inputs_norm, output, λ, inputs_test, inputs_test_norm, λ_Re180)
    plot!(1:length(l), l, label = "$key ($t s, pmax = $(maximum(p)))")
    display(current())
end

# Activation function analysis
act_options = Dict("relu" => relu,"sigmoid" => sigmoid, "tanh" => tanh, "softplus" => softplus, "leakyrelu" => leakyrelu)
act_analysis = plot()

for key in keys(act_options)
    l,t,p = train(AdaMax(0.01), act_options[key], 500, 300, 100,Flux.mse, inputs, inputs_norm, output, λ, inputs_test, inputs_test_norm, λ_Re180)
    plot!(1:length(l), l, label = "$key ($t s, pmax = $(maximum(p)))")
    display(current())
end

# Loss function analysis
loss_options = Dict("mse" => Flux.mse,"mae" => Flux.mae, "huber" => Flux.huber_loss)
loss_analysis = plot()

for key in keys(loss_options)
    l,t,p = train(ADAM(), leakyrelu, 1000, 300, 100,loss_options[key],inputs, inputs_norm, output, λ, inputs_test, inputs_test_norm, λ_Re180)
    plot!(1:length(l), l, label = "$key ($t s, pmax = $(maximum(p)))")
    display(current())
end

# Number of nodes analysis
node_options = [10,100,1000]
node_analysis = plot()

for n_nodes in node_options
    l,t,p = train(ADAM(), leakyrelu, 1000, 1000, n_nodes,Flux.mse, inputs, inputs_norm, output, λ, inputs_test, inputs_test_norm, λ_Re180)
    plot!(1:length(l), l, label = "$n_nodes ($t s)", xlabel = "Epochs", ylabel = "Residual Loss")
    display(current())
end