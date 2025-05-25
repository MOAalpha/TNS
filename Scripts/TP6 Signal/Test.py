import matplotlib.pyplot as plt
from matplotlib.table import Table

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Table data (example data - replace with your actual values)
data = [
    ["zéros/pôles", "Hz1a", "Hz1b", "Hz1c", "Hz2a", "Hz2b", "Hz2c"],
    ["z_p", "0.50+0.86j", "0.71+0.71j", "1.00+0.00j", "0.87+0.50j", "0.50+0.87j", "0.00+1.00j"],
    ["|z_p|", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["arg[z_p]", "1.0472", "0.7854", "0.0000", "0.5236", "1.0472", "1.5708"],
    ["v_p", "0.1667", "0.1250", "0.0000", "0.0833", "0.1667", "0.2500"],
    ["p_q", "0.60+0.40j", "0.50+0.50j", "0.80+0.00j", "0.70+0.30j", "0.40+0.60j", "0.00+0.90j"],
    ["|p_q|", "0.7211", "0.7071", "0.8000", "0.7616", "0.7211", "0.9000"],
    ["arg[p_q]", "0.5880", "0.7854", "0.0000", "0.4049", "0.9828", "1.5708"],
    ["v_q", "0.0936", "0.1250", "0.0000", "0.0644", "0.1564", "0.2500"]
]

# Create table
table = ax.table(cellText=data,
                loc='center',
                cellLoc='center',
                colWidths=[0.15] + [0.14]*6)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style header row
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#4F81BD')
        cell.set_text_props(color='white', weight='bold')
    elif i % 2 == 0:
        cell.set_facecolor('#D3DFEE')
    else:
        cell.set_facecolor('#E9E9E9')
    cell.set_edgecolor('white')

plt.title("Table of Filter Poles and Zeros Characteristics", pad=20)
plt.tight_layout()
plt.savefig('filter_table.png', dpi=300, bbox_inches='tight')
plt.show()