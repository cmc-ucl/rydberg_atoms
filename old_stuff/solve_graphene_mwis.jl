using JSON, UnitDiskMapping, Graphs, GenericTensorNetworks, LinearAlgebra

graphene_coordinate_weights = JSON.parsefile("graphene_coordinate_weights_mwis.json")
num_atoms = length(graphene_coordinate_weights[1])
Rb = 6.1
locs = [1e6 .* [graphene_coordinate_weights[1][i], graphene_coordinate_weights[2][i]] for i in 1 : num_atoms] 
g = unit_disk_graph(locs, Rb)
# show_graph(g; locs = locs./Rb, vertex_colors = ["white" for i in 1:nv(g)])



weights = graphene_coordinate_weights[3] ; # weights needs to be positive, uniform here for MIS
MWIS = collect(Int, solve(IndependentSet(g; weights=weights), SingleConfigMax())[].c.data) ; 
# println(MWIS, " ", sum(MWIS))
vertex_colors = [MWIS[i]==1 ? "red" : "white" for i in 1 : num_atoms]
# show_graph(g, locs = locs/Rb, vertex_colors=vertex_colors)

push!(graphene_coordinate_weights, MWIS)

open("graphene_coordinate_weights_mwis.json","w") do f 
    write(f, JSON.json(graphene_coordinate_weights)) 
end
