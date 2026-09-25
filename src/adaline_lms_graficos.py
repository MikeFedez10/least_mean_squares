import os
import matplotlib.pyplot as plt
import numpy as np

# Configurar ruta a la carpeta 'figures' (un nivel arriba de 'src')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def train_lms(X, d, initial_weights, mu, target_xi=0.1, max_epochs=300):
    """Entrena un modelo Adaline usando la regla de aprendizaje LMS (Paso más descendente).

    Retorna el vector de pesos finales, el historial de error cuadrático medio y
    las épocas.
    """
    w = initial_weights.copy()
    xi_history = []
    epoch = 0

    # Error promedio inicial (xi_p)
    y_init = np.dot(X, w)
    e_init = d - y_init
    xi_p = np.mean(e_init**2)
    xi_history.append(xi_p)

    # Ciclo de entrenamiento patrón por patrón
    while xi_p > target_xi and epoch < max_epochs:
        for k in range(len(X)):
            x_k = X[k]
            d_k = d[k]
            y_k = np.dot(w, x_k)
            e_k = d_k - y_k

            # Regla del paso más descendente
            w = w + 2 * mu * e_k * x_k

        # Error cuadrático medio de la época
        y_epoch = np.dot(X, w)
        e_epoch = d - y_epoch
        xi_p = np.mean(e_epoch**2)
        xi_history.append(xi_p)
        epoch += 1

    return w, xi_history, epoch


def plot_gate_convergence(
    gate_name, d, X, mus, initial_weights, target_xi=0.1, max_epochs=300
):
    """Genera la figura de 2x2 subgráficas para una compuerta lógica específica."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Compuerta {gate_name} - Evolución de $\\xi_p$ para cada valor de $\\mu$",
        fontsize=14,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    for i, mu in enumerate(mus):
        _, xi_history, epoch = train_lms(
            X, d, initial_weights, mu, target_xi, max_epochs
        )

        ax = axes_flat[i]
        ax.plot(
            range(len(xi_history)),
            xi_history,
            color="#1f77b4",
            linewidth=1.8,
            label=f"$\\xi_p$ (Épocas: {epoch})",
        )
        ax.axhline(
            y=target_xi,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Meta $\\xi_p = {target_xi}$",
        )

        ax.set_title(f"Tasa de aprendizaje $\\mu = {mu}$", fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("$\\xi_p$")
        ax.set_ylim(bottom=-0.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    plt.tight_layout()

    # Guardar automáticamente la figura en la carpeta 'figures'
    filepath = os.path.join(FIGURES_DIR, f"compuerta_{gate_name.lower()}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    # 1. Matriz de Entradas X [x0, x1, x2] con bias x0 = 1
    X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    # 2. Salidas deseadas 'd'
    gates = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "NOR": np.array([1, 0, 0, 0]),
        "XOR": np.array([0, 1, 1, 0]),
    }

    mus = [0.01, 0.25, 0.50, 0.75]
    target_xi = 0.1
    max_epochs = 300

    # Vector de pesos iniciales compartidos
    np.random.seed(42)
    initial_weights = np.random.rand(3) * 0.001

    # 3. Iteración y graficado por compuerta
    for gate_name, d in gates.items():
        plot_gate_convergence(
            gate_name, d, X, mus, initial_weights, target_xi, max_epochs
        )


if __name__ == "__main__":
    main()