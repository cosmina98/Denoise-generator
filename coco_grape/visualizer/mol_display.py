import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from coco_grape.visualizer.display import draw_graphs

def nx_to_rdkit(graph):
    """Convert a NetworkX graph to an RDKit molecule.
    
    Args:
        graph (networkx.Graph): A NetworkX graph representing the molecule, 
                                 where nodes have 'label' attribute for atom types,
                                 and edges have 'label' attribute for bond types.
    
    Returns:
        Chem.Mol: An RDKit molecule object created from the input graph.
    """
    m = Chem.MolFromSmiles('')  # Create an empty RDKit molecule
    mw = Chem.RWMol(m)  # Initialize a writable version of the RDKit molecule
    atom_index = {}  # Dictionary to map node indices to RDKit atom indices
    
    # Iterate over all nodes in the graph
    for n, d in graph.nodes(data=True):
        atom_index[n] = mw.AddAtom(Chem.Atom(d['label']))  # Add an atom to the RDKit molecule and store its index

    # Iterate over all edges in the graph
    for a, b, d in graph.edges(data=True):
        start = atom_index[a]  # Get the index of the starting atom
        end = atom_index[b]  # Get the index of the ending atom
        bond_type = d.get("label", '1')  # Get the bond type, default to single if not specified
        
        # Add the appropriate bond type between the two atoms
        if bond_type == '1':
            mw.AddBond(start, end, Chem.BondType.SINGLE)  # Add a single bond
        elif bond_type == '2':
            mw.AddBond(start, end, Chem.BondType.DOUBLE)  # Add a double bond
        elif bond_type == '3':
            mw.AddBond(start, end, Chem.BondType.TRIPLE)  # Add a triple bond
        # more options can be found at the RDKit documentation link provided below
        # http://www.rdkit.org/Python_Docs/rdkit.Chem.rdchem.BondType-class.html
        else:
            raise Exception('bond type not implemented')  # Raise an exception for unsupported bond types

    mol = mw.GetMol()  # Finalize the molecule construction by obtaining the immutable molecule
    return mol  # Return the constructed RDKit molecule


def set_coordinates(compounds):
    """Set 2D coordinates for a list of RDKit molecule objects.
    
    Args:
        compounds (list): A list of RDKit molecule objects that need coordinates set.
    
    Raises:
        Exception: If any molecule in the list is None, an exception is raised indicating failure.
    """
    # Iterate over each molecule in the provided list of compounds
    for m in compounds:
        if m:  # Check if the molecule is not None
            # Update the property cache to avoid "RuntimeError: Pre-condition Violation"
            m.UpdatePropertyCache(strict=False)
            
            # Compute the 2D coordinates for the molecule
            AllChem.Compute2DCoords(m)
        else:
            # Raise an exception if a None molecule is found
            raise Exception('''set coordinates failed..''')  # Indicate failure in setting coordinates


def get_smiles_strings(graphs):
    compounds = map(nx_to_rdkit, graphs)


def nx_to_image(graphs, n_graphs_per_line=5, size=250, title_key=None, titles=None):
    """Convert a list of NetworkX graphs to an image representation.
    
    Args:
        graphs (list or nx.Graph): A list of NetworkX graph objects or a single graph.
        n_graphs_per_line (int): Number of graphs to display per line in the image.
        size (int): Size of each individual graph's image.
        title_key (str, optional): Key for fetching titles from graph attributes.
        titles (list, optional): Custom list of titles for each graph.
        
    Raises:
        Exception: If a single graph is provided instead of a list.
    
    Returns:
        Image: An image object containing the rendered graphs.
    """
    
    # Check if the input is a single NetworkX graph; raise an exception if so
    if isinstance(graphs, nx.Graph):
        raise Exception("give me a list of graphs")
    
    # Convert NetworkX graphs to RDKit molecule objects using a mapping function
    compounds = list(map(nx_to_rdkit, graphs))
    
    # Handle the subtitles for each graph based on available information
    if title_key:
        # Extract titles from the graph's attributes using the specified title key
        legend = [g.graph.get(title_key, 'N/A') for g in graphs]
    elif titles:
        # Use the provided custom titles directly
        legend = titles
    else:
        # Default to numbering the graphs if no titles are available
        legend = [str(i) for i in range(len(graphs))]
    
    # Generate and return the image representation of the compounds
    return compounds_to_image(compounds, n_graphs_per_line=n_graphs_per_line, size=size, legend=legend)


def compounds_to_image(compounds, n_graphs_per_line=5, size=250, legend=None):
    # calculate coordinates:
    set_coordinates(compounds)
    # make the image
    return Draw.MolsToGridImage(compounds, molsPerRow=n_graphs_per_line, subImgSize=(size, size), legends=legend)

def draw_molecules(graphs, titles=None, num=None, n_graphs_per_line=7, size=7):
    """Draw a set of molecular graphs and display them as images.
    
    Args:
        graphs (list): A list of NetworkX graph objects representing molecules.
        titles (list, optional): Custom titles for each molecule; defaults to indices.
        num (int, optional): The number of graphs to display; if None, all are shown.
        n_graphs_per_line (int): Number of graphs to display per line in the output.
        size (int): Size of each individual graph's image (not used directly).
        
    Returns:
        None: Displays the images of the graphs.
    """
    
    # If no titles are provided, generate default titles based on the index of each graph
    if titles is None:
        titles = [str(i) for i in range(len(graphs))]
    
    # Limit the graphs and titles to the specified number if 'num' is provided
    if num is not None:
        gs = graphs[:num]  # Select only the first 'num' graphs
        titles = titles[:num]  # Select corresponding titles
    else:
        gs = graphs  # Use all graphs if 'num' is not specified
    
    # Assign each graph an 'id' attribute based on its title
    for g, t in zip(gs, titles):
        g.graph['id'] = str(t)
    
    try:
        # Loop through the graphs in chunks of 50 for rendering
        for i in range(0, len(gs), 50):
            # Create an image from the current chunk of graphs and display it
            img = nx_to_image(gs[i:i+50], n_graphs_per_line=n_graphs_per_line, titles=titles[i:i+50])
            display(img)  # Display the generated image
    except Exception as e:
        # Fallback to drawing graphs using a different method in case of an error
        draw_graphs(gs, titles=titles, n_graphs_per_line=n_graphs_per_line, size=size)
