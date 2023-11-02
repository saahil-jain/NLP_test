import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [37.90094470217938, 39.08655177849748, 39.68671274962551, 39.887653994474135, 39.80247953348627]

x_label = "Beam Size"
y_label = "BLEU Score"

plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f'{y_label} vs {x_label}')

plt.savefig('beam_plot.png')