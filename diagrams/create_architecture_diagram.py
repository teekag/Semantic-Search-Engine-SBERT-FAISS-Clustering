"""
Script to create a system architecture diagram for the Semantic Search Engine.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# Set up the figure
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Define colors
colors = {
    'background': '#f5f5f5',
    'box': '#e1f5fe',
    'box_border': '#0288d1',
    'arrow': '#0288d1',
    'text': '#333333',
    'highlight': '#ff5722',
    'section': '#e8f5e9',
    'section_border': '#43a047'
}

# Set background color
ax.set_facecolor(colors['background'])

# Create the main section boxes
sections = [
    {'name': 'Document Processing', 'x': 0.5, 'y': 5, 'width': 9, 'height': 1.5, 'color': colors['section'], 'border': colors['section_border']},
    {'name': 'Vector Search', 'x': 0.5, 'y': 2.5, 'width': 9, 'height': 2, 'color': colors['section'], 'border': colors['section_border']},
    {'name': 'Analysis & Visualization', 'x': 0.5, 'y': 0.5, 'width': 9, 'height': 1.5, 'color': colors['section'], 'border': colors['section_border']}
]

for section in sections:
    rect = patches.Rectangle(
        (section['x'], section['y']), 
        section['width'], 
        section['height'], 
        linewidth=2, 
        edgecolor=section['border'], 
        facecolor=section['color'], 
        alpha=0.3,
        zorder=1
    )
    ax.add_patch(rect)
    ax.text(
        section['x'] + 0.2, 
        section['y'] + section['height'] - 0.3, 
        section['name'], 
        fontsize=14, 
        fontweight='bold', 
        color=colors['text'],
        zorder=2
    )

# Create component boxes
components = [
    # Document Processing
    {'name': 'Raw Text\nDocuments', 'x': 1, 'y': 5.3, 'width': 1.5, 'height': 0.8, 'section': 0},
    {'name': 'Text\nPreprocessing', 'x': 3.2, 'y': 5.3, 'width': 1.5, 'height': 0.8, 'section': 0},
    {'name': 'SBERT\nEmbedding', 'x': 5.4, 'y': 5.3, 'width': 1.5, 'height': 0.8, 'section': 0},
    {'name': 'Document\nVectors', 'x': 7.6, 'y': 5.3, 'width': 1.5, 'height': 0.8, 'section': 0},
    
    # Vector Search
    {'name': 'FAISS\nIndex', 'x': 1.5, 'y': 3.2, 'width': 1.5, 'height': 0.8, 'section': 1},
    {'name': 'Query\nEmbedding', 'x': 3.7, 'y': 3.2, 'width': 1.5, 'height': 0.8, 'section': 1},
    {'name': 'Similarity\nSearch', 'x': 5.9, 'y': 3.2, 'width': 1.5, 'height': 0.8, 'section': 1},
    {'name': 'Ranked\nResults', 'x': 8.1, 'y': 3.2, 'width': 1.5, 'height': 0.8, 'section': 1},
    
    # Analysis & Visualization
    {'name': 'Clustering\nAlgorithms', 'x': 2, 'y': 0.8, 'width': 1.5, 'height': 0.8, 'section': 2},
    {'name': 'Dimensionality\nReduction', 'x': 4.2, 'y': 0.8, 'width': 1.5, 'height': 0.8, 'section': 2},
    {'name': 'Interactive\nVisualizations', 'x': 6.4, 'y': 0.8, 'width': 1.5, 'height': 0.8, 'section': 2}
]

for i, comp in enumerate(components):
    rect = patches.Rectangle(
        (comp['x'], comp['y']), 
        comp['width'], 
        comp['height'], 
        linewidth=1.5, 
        edgecolor=colors['box_border'], 
        facecolor=colors['box'], 
        alpha=0.8,
        zorder=3
    )
    ax.add_patch(rect)
    ax.text(
        comp['x'] + comp['width']/2, 
        comp['y'] + comp['height']/2, 
        comp['name'], 
        fontsize=10, 
        ha='center', 
        va='center', 
        color=colors['text'],
        zorder=4
    )

# Add arrows between components
arrows = [
    # Document Processing flow
    {'start': 0, 'end': 1},
    {'start': 1, 'end': 2},
    {'start': 2, 'end': 3},
    
    # Vector Search flow
    {'start': 4, 'end': 6},
    {'start': 5, 'end': 6},
    {'start': 6, 'end': 7},
    
    # Analysis flow
    {'start': 8, 'end': 9},
    {'start': 9, 'end': 10},
    
    # Cross-section flows
    {'start': 3, 'end': 4, 'curved': True},
    {'start': 3, 'end': 8, 'curved': True}
]

for arrow in arrows:
    start_comp = components[arrow['start']]
    end_comp = components[arrow['end']]
    
    # Calculate start and end points
    if 'curved' in arrow and arrow['curved']:
        # For cross-section arrows
        if start_comp['section'] != end_comp['section']:
            # Vertical arrows between sections
            start_x = start_comp['x'] + start_comp['width']/2
            start_y = start_comp['y']
            
            end_x = end_comp['x'] + end_comp['width']/2
            end_y = end_comp['y'] + end_comp['height']
            
            # Create a curved path
            verts = [
                (start_x, start_y),  # start point
                (start_x, (start_y + end_y)/2),  # control point
                (end_x, (start_y + end_y)/2),  # control point
                (end_x, end_y)  # end point
            ]
            
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4
            ]
            
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor=colors['arrow'], linewidth=1.5, zorder=2)
            ax.add_patch(patch)
            
            # Add arrow head
            arrow_head_size = 0.1
            dx = 0
            dy = arrow_head_size
            ax.arrow(end_x, end_y + 0.01, dx, dy, head_width=0.1, head_length=0.1, fc=colors['arrow'], ec=colors['arrow'], zorder=2)
    else:
        # For straight arrows within sections
        start_x = start_comp['x'] + start_comp['width']
        start_y = start_comp['y'] + start_comp['height']/2
        
        end_x = end_comp['x']
        end_y = end_comp['y'] + end_comp['height']/2
        
        ax.arrow(
            start_x, start_y, 
            end_x - start_x - 0.1, end_y - start_y, 
            head_width=0.1, head_length=0.1, 
            fc=colors['arrow'], ec=colors['arrow'],
            zorder=2
        )

# Add title
plt.text(5, 6.7, 'Semantic Search Engine Architecture', fontsize=18, fontweight='bold', ha='center', color=colors['text'])

# Add legend for components
legend_elements = [
    patches.Patch(facecolor=colors['box'], edgecolor=colors['box_border'], label='Component'),
    patches.Patch(facecolor=colors['section'], edgecolor=colors['section_border'], alpha=0.3, label='Processing Stage')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True)

# Save the figure
plt.tight_layout()
plt.savefig('diagrams/system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Architecture diagram created and saved to diagrams/system_architecture.png")
