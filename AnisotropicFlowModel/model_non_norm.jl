# Train a model based on the data gathering from the DNS and CFD
using DataFrames, CSV, Random, Flux
using .RS
using Plots
using Statistics
using StatsBase

# Functions
function gather(file, root)
    return DataFrame(CSV.File(joinpath(root, file)))
end

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

function export_results(y_plus, prediction, truth, name="RS_results_non_normalised")
    CSV.write(
        joinpath(pwd(),"Code", "Post Processing", "results",join([name,".csv"])),
        DataFrames.DataFrame(y_plus=y_plus, uu_yhat=prediction[1,:], vv_yhat=prediction[2,:], ww_yhat=prediction[3,:], uu_y=truth[1,:], vv_y=truth[2,:], ww_y=truth[3,:]))
end

base = joinpath(pwd(),"Code");

# Collate and shuffle the training data
Re180_training_data_in = gather("train_in_Re180.csv",joinpath(base, "CFD", "results"))
Re180_training_data_out = gather("train_out_Re180.csv",joinpath(base, "DNS", "results"))
Re180_training_input = hcat(Re180_training_data_in.y_plus,Re180_training_data_in.k,Re180_training_data_in.ω)
Re180_training_output = hcat(Re180_training_data_out.var_uu, Re180_training_data_out.var_vv, Re180_training_data_out.var_ww) 
Re180_λ = Re180_training_data_in.λ

training_data = mix(joinpath(base,"CFD","results"),joinpath(base,"DNS","results"))
training_input = hcat(training_data.y_plus,training_data.k,training_data.ω) # the input and lambda is normalised, but the target is not
training_output = hcat(training_data.var_uu, training_data.var_vv, training_data.var_ww)
λ = training_data.λ


function train(opt,act,epochs,batchsize, hidden_nodes,loss_function,training_input,training_output,λ,test, test_λ)
    # Create the data batch to be processed by the network
    data = Flux.DataLoader((training_input', λ, training_output'), batchsize = batchsize)

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
    loss(input, λ, comparison) = loss_function(RS.calculation(input,model(input),λ), comparison)

    # Optimiser and parameters
    prms = Flux.params(model) # retrieve the network params of the previously defined model

    # Training loop
    loss_history = [];
    time = @elapsed begin
    for epoch in 1:epochs
        #train model
        Flux.train!(loss, prms, data, opt)

        # Initiate loss function and store the residuals 
        train_loss = loss(training_input', λ, training_output')
        push!(loss_history, train_loss)
        println("Epoch = $epoch : Training Loss = $train_loss")
    end
    end
    # Return historic losses, time elapsed and final training predicition 
    return loss_history,time,RS.calculation(training_input',model(training_input'),λ),RS.calculation(test',model(test'),test_λ) #you cant call model() outside of this function becuse it finds a reminant model and is not true data
end

# Normal Run
optimiser = RMSProp()
activation_function = leakyrelu
loss_function = Flux.mse

epochs = 5000
batchsize = 300
hidden_nodes = 200

l_nn,t_nn,p_nn,p180_nn = train(optimiser, activation_function, epochs, batchsize, hidden_nodes, loss_function,training_input,training_output,λ,Re180_training_input, Re180_λ)
loss_plot = plot(1:length(l), l_nn, label = "Unnormalised Model Loss ($t_nn s, pmax = $(maximum(p_nn)))")
maximum(p_nn)
maximum(p180_nn)
p180_nn
export_results(Re180_training_input[:,1],p180_nn,Re180_training_output', "Re180")
export_results(training_data[:,1],p_nn,training_output')

# Optimiser analysis
opt_options = Dict("ADAM" => ADAM(),"Descent" => Descent(), "Momentum" => Momentum(), "AdaMax" => AdaMax(), "RMSProp" => RMSProp())
opt_analysis = plot()

for key in keys(opt_options)
    l,t,p = train(opt_options[key], leakyrelu, 1000, 200, 200,Flux.mse,training_input,training_output,λ,Re180_training_input, Re180_λ)
    plot!(1:length(l), l, label = "$key ($t s), pmax = $(maximum(p)))")
    display(current())
end

# Activation function analysis
act_options = Dict("relu" => relu,"sigmoid" => sigmoid, "tanh" => tanh, "softplus" => softplus, "leakyrelu" => leakyrelu)
act_analysis = plot()

for key in keys(act_options)
    l,t,P = train(ADAM(), act_options[key], 1000, 200, 200, Flux.mse,training_input,training_output,λ,Re180_training_input, Re180_λ)
    plot!(1:length(l), l, label = "$key ($t s), pmax = $(maximum(p)))")
    display(current())
end

# Batchsize analysis
bs_options = [100,1000]
bs_analysis = plot()

for batch_size in bs_options
    l,t = train(ADAM(), relu, 500, batch_size , 10,Flux.mse,training_input,training_output,λ,Re180_training_input, Re180_λ)
    plot!(1:length(l), l, label = "$batch_size ($t s)")
    display(current())
end

# Number of nodes analysis
node_options = [10,100,1000]
node_analysis = plot()

for n_nodes in node_options
    l,t = train(ADAM(), relu, 500, 50 , n_nodes,training_input,training_output,λ)
    plot!(1:length(l), l, label = "$n_nodes ($t s)", xlabel = "Epochs", ylabel = "Residual Loss")
    display(current())
end
