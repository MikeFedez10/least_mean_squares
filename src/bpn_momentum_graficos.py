import os
import matplotlib.pyplot as plt
import numpy as np

# Configurar ruta a la carpeta 'figures' (un nivel arriba de 'src')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def sigmoid(z):
    """Función de activación sigmoide unipolar."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(o):
    """Derivada de la sigmoide expresada en términos de la salida o."""
    return o * (1.0 - o)


def train_bpn(
    X,
    y_target,
    W1_init,
    W2_init,
    mu=0.5,
    alpha=0.1,
    target_mse=0.01,
    max_epochs=10000,
):
    """Entrena una red BPN (2-2-1) con o sin Momentum.

    W1: Pesos capa de entrada a oculta (3x2, incluye bias x0=1) W2: Pesos capa
    oculta a salida (3x1, incluye bias h0=1)
    """
    W1 = W1_init.copy()
    W2 = W2_init.copy()

    # Arreglos para almacenar el cambio de peso anterior (Momentum)
    dW1_prev = np.zeros_like(W1)
    dW2_prev = np.zeros_like(W2)

    mse_history = []
    num_samples = len(X)

    for epoch in range(max_epochs):
        # 1. Forward pass sobre todos los patrones para medir MSE
        # Capa oculta: net_h = X @ W1 -> a_h = sigmoide(net_h)
        net_h = np.dot(X, W1)
        a_h = sigmoid(net_h)

        # Agregar término de sesgo a la capa oculta [1, a_h1, a_h2]
        a_h_ext = np.hstack([np.ones((num_samples, 1)), a_h])

        # Capa de salida: net_o = a_h_ext @ W2 -> o_out = sigmoide(net_o)
        net_o = np.dot(a_h_ext, W2)
        o_out = sigmoid(net_o)

        # Cálculo del Error Cuadrático Medio (MSE)
        error_pattern = y_target - o_out
        mse = np.mean(error_pattern**2)
        mse_history.append(mse)

        # Verificar criterio de parada
        if mse <= target_mse:
            break

        # Acumuladores de gradientes para la época
        grad_W1 = np.zeros_like(W1)
        grad_W2 = np.zeros_like(W2)

        # 2. Reperpetuación patrón por patrón (online / batch gradient update)
        for p in range(num_samples):
            x_p = X[p : p + 1]  # Formato vector fila (1x3)
            y_p = y_target[p : p + 1]  # (1x1)

            # Re-evaluar forward para el patrón p
            net_h_p = np.dot(x_p, W1)
            a_h_p = sigmoid(net_h_p)
            a_h_ext_p = np.hstack([np.ones((1, 1)), a_h_p])

            net_o_p = np.dot(a_h_ext_p, W2)
            o_p = sigmoid(net_o_p)

            # Error en la unidad de salida: delta_o = (y - o) * o * (1 - o)
            delta_o = (y_p - o_p) * sigmoid_derivative(o_p)

            # Error en las unidades ocultas (excluyendo el bias de W2)
            # delta_h = (delta_o @ W2_nobias^T) * a_h * (1 - a_h)
            W2_nobias = W2[1:, :]  # (2x1)
            delta_h = np.dot(delta_o, W2_nobias.T) * sigmoid_derivative(a_h_p)

            # Acumular gradientes
            grad_W2 += np.dot(a_h_ext_p.T, delta_o)
            grad_W1 += np.dot(x_p.T, delta_h)

        # 3. Actualización de pesos con regla de Momentum
        dW2 = mu * grad_W2 + alpha * dW2_prev
        dW1 = mu * grad_W1 + alpha * dW1_prev

        W2 += dW2
        W1 += dW1

        # Almacenar cambios de peso para la siguiente época
        dW2_prev = dW2
        dW1_prev = dW1

    return W1, W2, mse_history, len(mse_history) - 1


def plot_bpn_comparison(gate_name, X, y_target, target_mse=0.01):
    """Genera y guarda la gráfica comparativa BPN Estándar vs. BPN con Momentum."""
    # Generar pesos iniciales compartidos (2-2-1)
    np.random.seed(42)
    W1_init = np.random.uniform(-0.5, 0.5, (3, 2))
    W2_init = np.random.uniform(-0.5, 0.5, (3, 1))

    # 1. Entrenamiento BPN Estándar (alpha = 0.1)
    W1_std, W2_std, mse_std, epochs_std = train_bpn(
        X,
        y_target,
        W1_init,
        W2_init,
        mu=0.5,
        alpha=0.1,
        target_mse=target_mse,
    )

    # 2. Entrenamiento BPN con Momentum (alpha = 0.9)
    W1_mom, W2_mom, mse_mom, epochs_mom = train_bpn(
        X,
        y_target,
        W1_init,
        W2_init,
        mu=0.5,
        alpha=0.9,
        target_mse=target_mse,
    )

    # Crear figura
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        range(len(mse_std)),
        mse_std,
        color="#1f77b4",
        linewidth=2,
        label=f"BPN Estándar ($\\alpha=0.1$) - Épocas: {epochs_std}",
    )
    plt.plot(
        range(len(mse_mom)),
        mse_mom,
        color="#ff7f0e",
        linewidth=2,
        label=f"BPN Momentum ($\\alpha=0.9$) - Épocas: {epochs_mom}",
    )

    plt.axhline(
        y=target_mse,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Meta $MSE = {target_mse}$",
    )

    plt.title(
        f"Compuerta {gate_name} - Comparación de Convergencia ($\\mu = 0.5$)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Época (epoch)", fontsize=11)
    plt.ylabel("Error Cuadrático Medio ($MSE$)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Guardar automáticamente la imagen
    filepath = os.path.join(
        FIGURES_DIR, f"compuerta_{gate_name.lower()}_bpn.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()

    # Imprimir resultados BPN Estándar
    print(
        f"\n--- Resultados Finales Compuerta {gate_name} (Estándar alpha=0.1) ---"
    )
    net_h_std = np.dot(X, W1_std)
    a_h_std = sigmoid(net_h_std)
    a_h_ext_std = np.hstack([np.ones((len(X), 1)), a_h_std])
    o_out_std = sigmoid(np.dot(a_h_ext_std, W2_std))

    for i in range(len(X)):
        x1, x2 = int(X[i, 1]), int(X[i, 2])
        deseado = int(y_target[i, 0])
        pred_val = o_out_std[i, 0]
        pred_bin = 1 if pred_val >= 0.5 else 0
        print(
            f"Entrada: [{x1}, {x2}] | Esperado: {deseado} | Salida Red: {pred_val:.4f} (Clase: {pred_bin})"
        )

    # Imprimir resultados BPN Momentum
    print(
        f"\n--- Resultados Finales Compuerta {gate_name} (Momentum alpha=0.9) ---"
    )
    net_h_mom = np.dot(X, W1_mom)
    a_h_mom = sigmoid(net_h_mom)
    a_h_ext_mom = np.hstack([np.ones((len(X), 1)), a_h_mom])
    o_out_mom = sigmoid(np.dot(a_h_ext_mom, W2_mom))

    for i in range(len(X)):
        x1, x2 = int(X[i, 1]), int(X[i, 2])
        deseado = int(y_target[i, 0])
        pred_val = o_out_mom[i, 0]
        pred_bin = 1 if pred_val >= 0.5 else 0
        print(
            f"Entrada: [{x1}, {x2}] | Esperado: {deseado} | Salida Red: {pred_val:.4f} (Clase: {pred_bin})"
        )


def main():
    # Matriz de entradas con Bias (x0 = 1)
    X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    gates = {
        "AND": np.array([[0], [0], [0], [1]]),
        "OR": np.array([[0], [1], [1], [1]]),
        "XOR": np.array([[0], [1], [1], [0]]),
    }

    for gate_name, y_target in gates.items():
        plot_bpn_comparison(gate_name, X, y_target)


if __name__ == "__main__":
    main()

#Machine Learning Models | ® Apizaco Institute of Technology | © 2026 Michael Fernández Sánchez. All rights reserved.