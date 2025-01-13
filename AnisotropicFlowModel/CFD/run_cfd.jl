#= Configure the variables, contrusct the mesh, 
check the mesh, run the simulation and then collate the data =#

# Variables
channel_height = 0.5
channel_length = 2.0

y_cells = 92 # this will be doubled because the volume is split in two
y_cells_stretch = 60

x_cells_front = 60 # Modify until the continuity looks good
x_cells_front_stretch = 50 # Modify until the continuity looks good
cells_x_back = 1000 # Modify until the continuity looks good

ν = 1e-5
k = 0.1
ω = 500
U∞ = 5

write_step = 100
max_iter = 5000

# Write configuration file

txt = [
    "// Mesh configuration\n",
    "channel_height     $channel_height;\n",
    "channel_length     $channel_length;\n",
    "\n",
    "cells_y    $y_cells; // total will be twice this number\n",
    "cells_y_stretch    $y_cells_stretch;\n",
    "\n",
    "cells_x_front   $x_cells_front; // modify until the grid continuity look ok\n",
    "cells_x_front_stretch   $x_cells_front_stretch; // modify until the grid continuity look ok\n",
    "\n",
    "cells_x_back    $cells_x_back; // modify until the grid continuity look ok\n",
    "\n",
    "// Flow and boundary configuration\n",
    "kinematic_viscosity	$ν;\n",
    "turb_kinetic_energy	$k;\n",
    "specific_dissipation_rate  $ω;\n",
    "freestream_velocity	$U∞; // x-component\n",
    "\n",
    "// Simulation controls\n",
    "write_step				$write_step;\n",
    "maximum_iterations	$max_iter;\n"
]

home = pwd()

function create_file(txt, name)
    file = open(joinpath("/home/esk/FoamCases/channel_turbulence_case", name), "w")
    for i in eachindex(txt)
        write(file, txt[i])
    end
    close(file)
end

function simpleFoam()
    cd("/home/esk/FoamCases/channel_turbulence_case")
    println(pwd())
    cmd = `simpleFoam`
    run(cmd)
end

function check_suitability()
    run(`clear`)

    output_file = open("yplus_check.txt", "w")
    redirect_stdout(output_file)
    cmd = `simpleFoam -postProcess -func yPlus -latestTime`
    run(cmd)
    close(output_file)
end

create_file(txt, "configuration")
simpleFoam()
check_suitability()
redirect_stdout(open("yplus_check.txt", "w"))
