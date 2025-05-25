import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal

# p1: Load the MATLAB file
mat_data = sio.loadmat(r'C:\Users\mathe\TNS\Sources\placement.mat')

# Extract filter coefficients
filters = {
    'Hz1a': {'num': mat_data['Hz1a_num'].flatten(), 'den': mat_data['Hz1a_den'].flatten()},
    'Hz1b': {'num': mat_data['Hz1b_num'].flatten(), 'den': mat_data['Hz1b_den'].flatten()},
    'Hz1c': {'num': mat_data['Hz1c_num'].flatten(), 'den': mat_data['Hz1c_den'].flatten()},
    'Hz2a': {'num': mat_data['Hz2a_num'].flatten(), 'den': mat_data['Hz2a_den'].flatten()},
    'Hz2b': {'num': mat_data['Hz2b_num'].flatten(), 'den': mat_data['Hz2b_den'].flatten()},
    'Hz2c': {'num': mat_data['Hz2c_num'].flatten(), 'den': mat_data['Hz2c_den'].flatten()},
}

# p2: Create transfer functions
for name in filters:
    filters[name]['tf'] = signal.TransferFunction(filters[name]['num'], filters[name]['den'])

# p3: Calculate frequency responses
frequencies = np.linspace(0, np.pi, 1000)
for name in filters:
    w, h = signal.freqz(filters[name]['num'], filters[name]['den'], worN=frequencies)
    filters[name]['freq_response'] = h
    filters[name]['freq_axis'] = w

# p4: Normalize gains
for name in filters:
    max_gain = np.max(np.abs(filters[name]['freq_response']))
    if max_gain > 0:
        filters[name]['freq_response_normalized'] = filters[name]['freq_response'] / max_gain
    else:
        filters[name]['freq_response_normalized'] = filters[name]['freq_response']


# p5: Plot pole-zero diagrams and frequency responses
def plot_filters(prefix, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title)

    colors = ['r', 'g', 'b']
    for i, suffix in enumerate(['a', 'b', 'c']):
        name = f'{prefix}{suffix}'
        tf = filters[name]['tf']

        # Pole-zero plot
        zeros = np.roots(tf.num)
        poles = np.roots(tf.den)

        # Plot unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)

        # Plot poles and zeros
        ax1.plot(np.real(zeros), np.imag(zeros), 'o', color=colors[i], label=f'Zeros {name}')
        ax1.plot(np.real(poles), np.imag(poles), 'x', color=colors[i], label=f'Poles {name}')

        # Frequency response
        ax2.plot(filters[name]['freq_axis'], np.abs(filters[name]['freq_response_normalized']),
                 color=colors[i], label=name)

    ax1.set_title('Pole-Zero Diagram')
    ax1.set_xlabel('Real')
    ax1.set_ylabel('Imaginary')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')

    ax2.set_title('Normalized Frequency Response')
    ax2.set_xlabel('Normalized Frequency [rad/sample]')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


plot_filters('Hz1', 'Series 1 Filters (Hz1a, Hz1b, Hz1c)')
plot_filters('Hz2', 'Series 2 Filters (Hz2a, Hz2b, Hz2c)')


# p6: Create table with zeros and poles information
def analyze_filter(name):
    tf = filters[name]['tf']
    zeros = np.roots(tf.num)
    poles = np.roots(tf.den)

    zero_info = []
    for z in zeros:
        zero_info.append({
            'z_p': z,
            '|z_p|': np.abs(z),
            'arg[z_p]': np.angle(z),
            'v_p': np.angle(z) / (2 * np.pi)
        })

    pole_info = []
    for p in poles:
        pole_info.append({
            'p_q': p,
            '|p_q|': np.abs(p),
            'arg[p_q]': np.angle(p),
            'v_q': np.angle(p) / (2 * np.pi)
        })

    return zero_info, pole_info


# Print table header
print("| zéros/pôles | Hz1a | Hz1b | Hz1c | Hz2a | Hz2b | Hz2c |")
print("|---|---|---|---|---|---|---|")

# For each row in the table, we'll collect the data
table_rows = [
    ("z_p", 'zero', lambda z: z['z_p']),
    ("|z_p|", 'zero', lambda z: z['|z_p|']),
    ("arg[z_p]", 'zero', lambda z: z['arg[z_p]']),
    ("v_p", 'zero', lambda z: z['v_p']),
    ("p_q", 'pole', lambda p: p['p_q']),
    ("|p_q|", 'pole', lambda p: p['|p_q|']),
    ("arg[p_q]", 'pole', lambda p: p['arg[p_q]']),
    ("v_q", 'pole', lambda p: p['v_q']),
]

for row_label, row_type, func in table_rows:
    row = f"| {row_label} |"
    for name in ['Hz1a', 'Hz1b', 'Hz1c', 'Hz2a', 'Hz2b', 'Hz2c']:
        zeros, poles = analyze_filter(name)

        if row_type == 'zero':
            data = zeros
        else:
            data = poles

        if data:
            val = func(data[0])  # Just show first zero/pole if multiple
            if isinstance(val, complex):
                row += f" {val:.2f} |"
            else:
                row += f" {val:.4f} |"
        else:
            row += " - |"
    print(row)

# p7: Determine filter characteristics
print("\nFilter Characteristics:")
for name in filters:
    tf = filters[name]['tf']
    order = max(len(tf.num) - 1, len(tf.den) - 1)

    # Determine filter type based on poles/zeros
    zeros = np.roots(tf.num)
    poles = np.roots(tf.den)

    # Check if all poles are inside unit circle (stability)
    stable = all(np.abs(poles) < 1) if len(poles) > 0 else True

    # Simple type detection (can be expanded)
    if len(zeros) == 0:
        ftype = "Low-pass"
    elif np.all(np.abs(zeros) == 0):  # All zeros at origin
        ftype = "Low-pass"
    elif np.all(np.abs(zeros) == 1):  # All zeros on unit circle
        if len(zeros) == 1 and zeros[0] == 1:
            ftype = "High-pass"
        else:
            ftype = "Band-stop"
    else:
        ftype = "Band-pass or other"

    print(f"{name}: Order {order}, {ftype}, {'Stable' if stable else 'Unstable'}")

# p8: Analyze gain evolution vs pole/zero placement
print("\nGain evolution analysis:")
print("The gain at any frequency v depends on the distance between the point M (e^(j2πv)) on the unit circle")
print("and the poles/zeros. The gain is calculated as:")
print("H(v) = product(distances from M to zeros) / product(distances from M to poles)")
print("Therefore:")
print("- Poles close to the unit circle increase gain at nearby frequencies")
print("- Zeros on the unit circle create complete attenuation at that frequency")
print("- Complex conjugate pole pairs create resonance peaks")
print("- Complex conjugate zero pairs create notches")
print("The exact relationship can be visualized by the pole-zero plots and frequency response curves above.")