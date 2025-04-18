import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from scipy.stats import binned_statistic
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from matplotlib.legend import Legend
import os
import subprocess

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

def load_data(base_dir, gse_file, pos_file, index_key_file, rescale):

    index_key = np.load(os.path.join(base_dir, index_key_file))
    posfile = pd.read_csv(pos_file, header=None)
    posfile_tuples = list(zip(posfile[1], posfile[0]))
    posfile_lookup = {tuple_: idx for idx, tuple_ in enumerate(posfile_tuples)}

    # Iterate over index_key and replace the third column based on posfile order
    for i in range(index_key.shape[0]):
        key_tuple = (index_key[i, 0], index_key[i, 1])
        if key_tuple in posfile_lookup:
            index_key[i, 2] = posfile_lookup[key_tuple]
    
    recursion = 0
    while not os.path.isfile(os.path.join(base_dir, gse_file)):
        print(str(os.path.join(base_dir, gse_file)) + ' not found.')
        base_dir += '/merged_fast/'
        index_key = index_key[np.load(os.path.join(base_dir, index_key_file))[:,1],:]
        #print(str(index_key))
        if recursion > 10: # assume not found, break
            break
        recursion += 1
        
    gse_data = np.loadtxt(os.path.join(base_dir, gse_file), delimiter=',')
    print('loaded ' + os.path.join(base_dir, gse_file) + ' ' + str(gse_data.shape))
    pos_data = np.loadtxt(pos_file, delimiter=',')[index_key[:,2],:]
    pos_data[:,2:] *= rescale/np.sqrt(np.sum(np.var(pos_data[:,2:],axis=0)))
    
    return gse_data, pos_data, index_key

def plot_data(data, title, color_data, ax, xlim, ylim):
    type_1_mask = color_data[:, 1] == 0
    type_2_mask = color_data[:, 1] == 1
    
    # Plot type-1 indices in black
    ax.scatter(data[type_1_mask, 0], data[type_1_mask, 1], c='black', s=2)
    
    # Plot type-2 indices with rainbow colors
    cmap = plt.cm.rainbow
    colors = cmap((color_data[type_2_mask, 2] - color_data[type_2_mask, 2].min()) / 
                  (color_data[type_2_mask, 2].max() - color_data[type_2_mask, 2].min()))
    ax.scatter(data[type_2_mask, 0], data[type_2_mask, 1], c=colors, s=2)
    
    ax.set_title(title)
    #ax.set_xlabel('X-coordinate')
    #ax.set_ylabel('Y-coordinate')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', 'box')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set major ticks at unit intervals but remove tick labels and ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)


def process_gse_data(gse_data, pos_data):
    pos_coords = pos_data[:, 2:]
    gse_coords = gse_data[:, 1:(1+pos_coords.shape[1])]
    
    pos_center = np.mean(pos_coords, axis=0)
    gse_center = np.mean(gse_coords, axis=0)
    pos_coords_centered = pos_coords - pos_center
    gse_coords_centered = gse_coords - gse_center
    
    # Calculate variance (summed across dimensions)
    pos_var = np.sum(np.var(pos_coords_centered, axis=0))
    gse_var = np.sum(np.var(gse_coords_centered, axis=0))
    
    # Calculate scale factor using square root of variance ratio
    scale_factor = np.sqrt(pos_var / gse_var)
    
    gse_coords_scaled = gse_coords_centered * scale_factor
    
    R, _ = orthogonal_procrustes(gse_coords_scaled, pos_coords_centered)
    
    transformed_gse = np.dot(gse_coords_centered, R) * scale_factor + pos_center
    
    return transformed_gse

def process_neighborhood(gse_coords, pos_coords):
    gse_center = np.mean(gse_coords, axis=0)
    pos_center = np.mean(pos_coords, axis=0)
    gse_centered = gse_coords - gse_center
    pos_centered = pos_coords - pos_center
    
    gse_var = np.sum(np.var(gse_centered, axis=0))
    pos_var = np.sum(np.var(pos_centered, axis=0))
    scale_factor = np.sqrt(pos_var / gse_var)
    
    gse_scaled = gse_centered * scale_factor
    
    R, _ = orthogonal_procrustes(gse_scaled, pos_centered)
    gse_transformed = np.dot(gse_scaled, R) + pos_center
    
    return gse_transformed

def calculate_rmsd(points1, points2):
    diff = points1 - points2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def analyze_neighborhoods(gse_data, pos_data, decoherence_sizes, n_samples=500):
    pos_coords = pos_data[:, 2:]
    gse_coords = gse_data[:, 1:(pos_coords.shape[1]+1)]

    results = []
    
    for deco_size in decoherence_sizes:
        rmsds = []
        spreads = []
        
        nn = NearestNeighbors(n_neighbors=deco_size, metric='euclidean')
        nn.fit(pos_coords)
        
        sample_indices = np.random.choice(len(pos_coords), n_samples, replace=False)
        
        for idx in sample_indices:
            _, indices = nn.kneighbors(pos_coords[idx].reshape(1, -1))
            indices = indices[0]
            
            gse_neighborhood = gse_coords[indices]
            pos_neighborhood = pos_coords[indices]
            
            gse_transformed = process_neighborhood(gse_neighborhood, pos_neighborhood)
            
            rmsd = calculate_rmsd(gse_transformed, pos_neighborhood)
            spread = np.mean(np.linalg.norm(pos_neighborhood - pos_neighborhood[0], axis=1))
            
            rmsds.append(rmsd)
            spreads.append(spread)
        
        results.append({
            'deco_size': deco_size,
            'rmsds': np.array(rmsds),
            'spreads': np.array(spreads)
        })
    
    return results


def plot_neighborhood_analysis(all_results, legend_dict=None, change_colorwheel=False):
    plt.figure(figsize=(12, 8))
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 16
    
    if change_colorwheel:
        colors = plt.cm.tab10(np.linspace(0, 1, change_colorwheel)[1:])
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (subdir, results), color in zip(all_results.items(), colors):
        all_spreads = np.concatenate([r['spreads'] for r in results])
        all_rmsds = np.concatenate([r['rmsds'] for r in results])
        
        # Use binned_statistic to consolidate points with similar x-values
        bin_means, bin_edges, _ = binned_statistic(all_spreads, all_rmsds, statistic='mean', bins=20)
        bin_stds, _, _ = binned_statistic(all_spreads, all_rmsds, statistic='std', bins=bin_edges)
        
        # Calculate the bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Use legend_dict if provided, otherwise use subdir
        label = legend_dict.get(subdir, subdir) if legend_dict else subdir
        
        # Plot the mean line
        plt.plot(bin_centers, bin_means, '-', color=color, label=label, linewidth=2)
        
        # Plot the shaded error band
        plt.fill_between(bin_centers, 
                         bin_means - bin_stds, 
                         bin_means + bin_stds, 
                         color=color, alpha=0.3)
    
    plt.xlabel('Mean Point Spread', fontsize=25)
    plt.ylabel('Mean RMSD', fontsize=25)
    plt.title('Neighborhood Analysis: RMSD vs Point Spread', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=25)
    
    # Create legend with adjusted box width
    legend = plt.legend(fontsize=16)
    
    # Adjust legend box width to fit text
    for text in legend.get_texts():
        text.set_bbox(dict(facecolor='none', edgecolor='none', pad=0))
    
    plt.tight_layout()
    plt.show()


def get_params(subdir):
    params_dict = {}

    # Open the file and read line by line
    with open(subdir + "/params.txt", "r") as file:
        for line in file:
            # Split the line into key and value based on whitespace
            key, value = line.split()
            # Store the key-value pair in the dictionary
            params_dict[key] = value
    return params_dict

def get_title_string(param_dict):
    title = list()
    for key in param_dict:
        if key == '-inference_eignum':
            title.append("$E$=" + str(param_dict[key]))
        elif key == '-sub_num':
            if int(param_dict[key]) > 0:
                title.append("$S$=" + str(param_dict[key]))
        elif key == '-sub_size':
            if int(param_dict[key]) > 0:
                title.append("$m_S$=" + str(param_dict[key]))
    return ', '.join(title)
    
    
def plotdirs(subdirs, rescale, plot_3d=False):
    # Load posfile.csv (assumed to be in the parent directory)
    pos_data = np.loadtxt('posfile.csv', delimiter=',')
    pos_data[:,2:] *= rescale/np.sqrt(np.sum(np.var(pos_data[:,2:],axis=0)))
    
    if not plot_3d:
       # Load posfile.csv (assumed to be in the parent directory)

        # Calculate axis limits
        x_min, x_max = np.floor(pos_data[:,2].min()) - 1, np.ceil(pos_data[:,2].max()) + 1
        y_min, y_max = np.floor(pos_data[:,3].min()) - 1, np.ceil(pos_data[:,3].max()) + 1
        xlim, ylim = (x_min, x_max), (y_min, y_max)

        # Calculate the number of rows and columns for the subplot grid
        n_plots = len(subdirs) + 1  # +1 for posfile.csv
        n_cols = 2  # or any other number you prefer
        n_rows = (n_plots - 1) // n_cols + 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 8*n_rows))
        axs = axs.flatten()

        # Plot posfile.csv data
        plot_data(pos_data[:, 2:4], 'Scatter plot of posfile.csv', pos_data, axs[0], xlim, ylim)

        # Process each subdirectory
        for i, subdir in enumerate(subdirs, start=1):
            
            if 'umap' in subdir:
                outfile='umap.txt'
            else:
                outfile='GSEoutput.txt'
            gse_data, pos_data, index_key = load_data(subdir, outfile, 'posfile.csv', 'index_key.npy', rescale)
            #print(index_key.shape)
            transformed_gse = process_gse_data(gse_data, pos_data)
            plot_data(transformed_gse, get_title_string(get_params(subdir)), pos_data, axs[i], xlim, ylim)

        # Remove any unused subplots
        for j in range(n_plots, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
    else:
        # 3D plotting
        fig = make_subplots(rows=len(subdirs)+1, cols=1, 
                            specs=[[{'type': 'scene'}] for _ in range(len(subdirs)+1)],
                            vertical_spacing=0.02)
        
        # Calculate the overall range for all data
        all_data = [pos_data[:, 2:5]]
        all_pos_data = [pos_data]
        for subdir in subdirs:
            if 'umap' in subdir:
                outfile='umap.txt'
            else:
                outfile='GSEoutput.txt'
            gse_data, pos_data, index_key = load_data(subdir, outfile, 'posfile.csv', 'index_key.npy', rescale)
            #print(index_key.shape)
            transformed_gse = process_gse_data(gse_data, pos_data)
            all_data.append(transformed_gse)
            all_pos_data.append(np.array(pos_data))
        
        all_data = np.vstack(all_data)
        min_vals = np.floor(np.min(all_data, axis=0))
        max_vals = np.ceil(np.max(all_data, axis=0))
        
        # Ensure unit grid spacing
        range_x = [min_vals[0], max_vals[0]]
        range_y = [min_vals[1], max_vals[1]]
        range_z = [min_vals[2], max_vals[2]]
        
        # Calculate aspect ratio
        x_range = range_x[1] - range_x[0]
        y_range = range_y[1] - range_y[0]
        z_range = range_z[1] - range_z[0]
        aspect_ratio = dict(x=x_range, y=y_range, z=z_range)
        
        # Plot posfile.csv data
        type_1_mask = pos_data[:, 1] == 0
        type_2_mask = pos_data[:, 1] == 1
        
        # Plot type-1 indices in black
        fig.add_trace(go.Scatter3d(x=pos_data[type_1_mask, 2], 
                                   y=pos_data[type_1_mask, 3], 
                                   z=pos_data[type_1_mask, 4],
                                   mode='markers',
                                   marker=dict(size=.75, color='black'),
                                   name='Type 1'),
                      row=1, col=1)
        
        # Plot type-2 indices with rainbow colors
        color_scale = pos_data[type_2_mask, 3]
        color_scale = (color_scale - color_scale.min()) / (color_scale.max() - color_scale.min())
        
        fig.add_trace(go.Scatter3d(x=pos_data[type_2_mask, 2], 
                                   y=pos_data[type_2_mask, 3], 
                                   z=pos_data[type_2_mask, 4],
                                   mode='markers',
                                   marker=dict(size=.75, color=color_scale, colorscale='rainbow'),
                                   name='Type 2'),
                      row=1, col=1)
        
        # Process each subdirectory
        for i, subdir in enumerate(subdirs, start=2):
            print([i,subdir])
            if 'umap' in subdir:
                outfile='umap.txt'
            else:
                outfile='GSEoutput.txt'
            gse_data, pos_data, index_key = load_data(subdir, outfile, 'posfile.csv', 'index_key.npy', rescale)
            type_1_mask = pos_data[:, 1] == 0
            type_2_mask = pos_data[:, 1] == 1
            color_scale = pos_data[type_2_mask, 3]
            color_scale = (color_scale - color_scale.min()) / (color_scale.max() - color_scale.min())
            transformed_gse = process_gse_data(gse_data,  pos_data)
            
            type_1_mask = index_key[:,0] == 0
            type_2_mask = index_key[:,0] == 1
            # Plot type-1 indices in black
            fig.add_trace(go.Scatter3d(x=transformed_gse[type_1_mask, 0], 
                                       y=transformed_gse[type_1_mask, 1], 
                                       z=transformed_gse[type_1_mask, 2],
                                       mode='markers',
                                       marker=dict(size=.75, color='black'),
                                       name=f'{subdir} Type 1'),
                          row=i, col=1)
            
            # Plot type-2 indices with rainbow colors
            fig.add_trace(go.Scatter3d(x=transformed_gse[type_2_mask, 0], 
                                       y=transformed_gse[type_2_mask, 1], 
                                       z=transformed_gse[type_2_mask, 2],
                                       mode='markers',
                                       marker=dict(size=.75, color=color_scale, colorscale='rainbow'),
                                       name=f'{subdir} Type 2'),
                          row=i, col=1)
        
        # Update layout for all subplots
        for i in range(1, len(subdirs)+2):
            fig.update_scenes(row=i, col=1,
                              xaxis=dict(range=range_x, dtick=1, showticklabels=False, title=''),
                              yaxis=dict(range=range_y, dtick=1, showticklabels=False, title=''),
                              zaxis=dict(range=range_z, dtick=1, showticklabels=False, title=''),
                              aspectmode='manual', aspectratio=aspect_ratio)
        
        # Adjust the layout to ensure all plots are visible without zooming out
        camera_settings = dict(eye=dict(x=3, y=0, z=17))
        layout_updates = {
            f'scene{i+1 if i > 0 else ""}': dict(
                xaxis=dict(range=range_x, dtick=1, showticklabels=False, title=''),
                yaxis=dict(range=range_y, dtick=1, showticklabels=False, title=''),
                zaxis=dict(range=range_z, dtick=1, showticklabels=False, title=''),
                aspectmode='manual', 
                aspectratio=aspect_ratio,
                camera=camera_settings
            ) for i in range(len(subdirs) + 1)
        }

        fig.update_layout(
            height=400*(len(subdirs)+1),  # Increased height
            width=600,  # Adjusted width
            title_text="3D Scatter Plots",
            margin=dict(l=0, r=0, t=30, b=0),  # Reduced margins
            **layout_updates
        )
        fig.show('browser')

# PLOTTING SCRIPT
dir = ''
change_colorwheel = False

os.chdir(dir)
subdirs = ['gse1/','gse4/','gse5/','gse7/']

for subdir in subdirs:
    print(subdir)
    print(subprocess.run("tail -10 " + subdir + "params.txt",shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout)
    
plotdirs(subdirs, rescale=4.0, plot_3d = True) # rescale value is hard-coded into simulation
print('done plotting')
# Neighborhood analysis for all subdirectories
pos_file = 'posfile.csv'
index_key_file = 'index_key.npy'
decoherence_sizes = [100, 200, 400, 800, 2000, 4000, 5000, 10000, 15000, 20000, 25000, 27000, 30000, 33000]

all_results = {}

for subdir in subdirs:
    print(f"Processing {subdir}")
    base_dir = os.path.join(dir, subdir)
    if subdir.startswith('gse') or  subdir.startswith('smle'):
        gse_file = 'GSEoutput.txt'
    else:
        gse_file = 'umap.txt'
    gse_data, pos_data, index_key = load_data(base_dir, gse_file, pos_file, index_key_file, rescale=4.0) # rescale value is hard-coded into simulation
    print(gse_data.shape)
    
    results = analyze_neighborhoods(gse_data, pos_data, decoherence_sizes)
    all_results[subdir] = results

plot_neighborhood_analysis(all_results, change_colorwheel)
