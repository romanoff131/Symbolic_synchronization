import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import adi
import random

# === Функция задания параметров Adalm Pluto SDR ===
def standart_settings(Pluto_IP="192.168.2.1", sample_rate=1e6, buffer_size=1e3, gain_mode="manual"):
    sdr = adi.Pluto(Pluto_IP)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.gain_control_mode_chan0 = gain_mode
    print(f"[INFO] SDR настроен: sample_rate={sample_rate}, buffer_size={buffer_size}, gain_mode={gain_mode}")
    return sdr

# === Функция отрисовки графиков ===
def plot_constellation(symbols, title="Constellation Diagram"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(np.real(symbols), np.imag(symbols), color='blue', s=50, marker='o', edgecolors='k', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.axhline(0, color='red', linewidth=1, linestyle='--')
    ax.axvline(0, color='red', linewidth=1, linestyle='--')
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    max_val = np.max(np.abs(symbols)) * 1.2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    plt.tight_layout()

# === Функция алгоритма фазовой синхронизации phase-locked-loop ===
def PLL(signal, constellation, mu=0.05, verbose=True):
    output = np.zeros_like(signal, dtype=np.complex128)
    theta = 0.0
    phase_error = np.zeros(len(signal), dtype=float)
    for n, x in enumerate(signal):
        # Remove estimated carrier phase
        x_rot = x * np.exp(-1j * theta)
        # Decision: map to nearest reference
        idx = np.argmin(np.abs(constellation - x_rot))
        decision = constellation[idx]
        # Calculate phase difference between received and reference
        err = np.angle(x_rot * np.conj(decision))
        phase_error[n] = err
        # Loop update
        theta += mu * err
        output[n] = x * np.exp(-1j * theta)
    if verbose:
        print(f"[INFO] PLL-QAM завершён. финальное theta={theta:.3f}, mean error={np.mean(phase_error):.3e}")
    return output

def gardner_ted_error(signal, nsp=10):
    error = np.zeros(len(signal) // nsp)
    for i in range(nsp, len(signal) - nsp, nsp):
        s_early = signal[i - nsp // 2]
        s_mid   = signal[i]
        s_late  = signal[i + nsp // 2]
        err = np.real((s_late - s_early) * np.conj(s_mid))
        error[i // nsp] = err
    print("[INFO] Gardner TED: ошибка БЕЗ синхронизации рассчитана.")
    return error

# === Функция алгоритма символьной синхронизации Gardner TED ===
def gardner_ted(signal, nsp=10, alpha=0.05):
    error = np.zeros(len(signal) // nsp)
    index_sync = []
    ema_error = 0  # экспоненциальное сглаживание для затухания

    for i in range(nsp, len(signal) - nsp, nsp):
        s_early = signal[i - nsp // 2]
        s_mid   = signal[i]
        s_late  = signal[i + nsp // 2]
        err = np.real((s_late - s_early) * np.conj(s_mid))
        if i == nsp:
            ema_error = err
        else:
            ema_error = alpha * err + (1 - alpha) * ema_error  # затухающее среднее
        error[i // nsp] = ema_error
        index_sync.append(i)

    print("[INFO] Gardner TED завершён.")
    return np.array(index_sync), error

# --- Функция для частотной коррекции через автокорреляцию ---
def autocorr_freq_correct(signal, n_lag=1):
    x = signal
    auto = x[n_lag:] * np.conj(x[:-n_lag])
    mean_phase_offset = np.angle(np.mean(auto))
    freq_error = mean_phase_offset / n_lag
    print(f"[INFO] Оценена частотная ошибка: {freq_error:.6f} rad/sample ({freq_error*180/np.pi:.3f} deg/sample)")
    n = np.arange(len(x))
    corrected = x * np.exp(-1j * freq_error * n)
    return corrected, freq_error

# === Функция генерации рандомной битовой последовательности ===
def randomDataGenerator(size):
    bits = [random.randint(0, 1) for _ in range(size)]
    print(f"[INFO] Сгенерировано {len(bits)} бит.")
    return bits

# === Функция модуляции QAM16 === 
def QAM16(bit_mass):
    qam16_table = {
        (0, 0, 0, 0): complex(-3, -3) / np.sqrt(10),
        (0, 0, 0, 1): complex(-3, -1) / np.sqrt(10),
        (0, 0, 1, 0): complex(-3,  3) / np.sqrt(10),
        (0, 0, 1, 1): complex(-3,  1) / np.sqrt(10),
        (0, 1, 0, 0): complex(-1, -3) / np.sqrt(10),
        (0, 1, 0, 1): complex(-1, -1) / np.sqrt(10),
        (0, 1, 1, 0): complex(-1,  3) / np.sqrt(10),
        (0, 1, 1, 1): complex(-1,  1) / np.sqrt(10),
        (1, 0, 0, 0): complex( 3, -3) / np.sqrt(10),
        (1, 0, 0, 1): complex( 3, -1) / np.sqrt(10),
        (1, 0, 1, 0): complex( 3,  3) / np.sqrt(10),
        (1, 0, 1, 1): complex( 3,  1) / np.sqrt(10),
        (1, 1, 0, 0): complex( 1, -3) / np.sqrt(10),
        (1, 1, 0, 1): complex( 1, -1) / np.sqrt(10),
        (1, 1, 1, 0): complex( 1,  3) / np.sqrt(10),
        (1, 1, 1, 1): complex( 1,  1) / np.sqrt(10),
    }
    if len(bit_mass) % 4 != 0:
        print("QAM16: Ошибка – длина битовой последовательности должна быть кратна 4.")
        raise Exception("QAM16: Длина битовой последовательности должна быть кратна 4.")
    sample = np.array([qam16_table[tuple(bit_mass[i:i+4])] for i in range(0, len(bit_mass), 4)])
    print(f"[INFO] QAM16: Сформировано {len(sample)} символов.")
    return sample

# === Функция модуляции QAM64 === 
def QAM64(bit_mass):
    ampl = 2**14
    if len(bit_mass) % 6 != 0:
        raise Exception("QAM64: Длина битовой последовательности должна быть кратна 6.")
    symbols = np.zeros(len(bit_mass) // 6, dtype=np.complex128)
    for i in range(0, len(bit_mass), 6):
        real_index = bit_mass[i] * 4 + bit_mass[i+1] * 2 + bit_mass[i+2]
        imag_index = bit_mass[i+3] * 4 + bit_mass[i+4] * 2 + bit_mass[i+5]
        real_val = 2 * real_index - 7
        imag_val = 2 * imag_index - 7
        symbols[i // 6] = complex(real_val, imag_val)
    print(f"[INFO] QAM64: Сформировано {len(symbols)} символов.")
    return symbols / np.sqrt(42) * ampl

# === TX/RX ===
def tx_signal(sdr, tx_lo, gain_tx, data, tx_cycle=True):
    sdr.tx_lo = int(tx_lo)
    sdr.tx_hardwaregain_chan0 = gain_tx
    sdr.tx_cyclic_buffer = tx_cycle
    print(f"[INFO] Передача сигнала: tx_lo={tx_lo}, gain_tx={gain_tx}")
    sdr.tx(data)

def rx_signal(sdr, rx_lo, gain_rx, cycle):
    sdr.rx_lo = int(rx_lo)
    sdr.rx_hardwaregain_chan0 = gain_rx
    data = []
    for _ in range(cycle):
        rx = sdr.rx()
        data.append(rx)
    print(f"[INFO] Принято {cycle} блоков данных от SDR.")
    return np.concatenate(data)

def get_constellation(bits_per_symbol):
    """
    Получение массива точек созвездия для нужной модуляции.
    """
    if bits_per_symbol == 4:
        qam16_table = {
            (0, 0, 0, 0): complex(-3, -3) / np.sqrt(10),
            (0, 0, 0, 1): complex(-3, -1) / np.sqrt(10),
            (0, 0, 1, 0): complex(-3,  3) / np.sqrt(10),
            (0, 0, 1, 1): complex(-3,  1) / np.sqrt(10),
            (0, 1, 0, 0): complex(-1, -3) / np.sqrt(10),
            (0, 1, 0, 1): complex(-1, -1) / np.sqrt(10),
            (0, 1, 1, 0): complex(-1,  3) / np.sqrt(10),
            (0, 1, 1, 1): complex(-1,  1) / np.sqrt(10),
            (1, 0, 0, 0): complex( 3, -3) / np.sqrt(10),
            (1, 0, 0, 1): complex( 3, -1) / np.sqrt(10),
            (1, 0, 1, 0): complex( 3,  3) / np.sqrt(10),
            (1, 0, 1, 1): complex( 3,  1) / np.sqrt(10),
            (1, 1, 0, 0): complex( 1, -3) / np.sqrt(10),
            (1, 1, 0, 1): complex( 1, -1) / np.sqrt(10),
            (1, 1, 1, 0): complex( 1,  3) / np.sqrt(10),
            (1, 1, 1, 1): complex( 1,  1) / np.sqrt(10),
        }
        return np.array(list(qam16_table.values()))
    elif bits_per_symbol == 6:
        decision_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
        # 64 точки, нормированный уровень!
        return np.array([complex(i, q) for i in decision_levels for q in decision_levels])
    else:
        raise ValueError("Unsupported bits_per_symbol for constellation!")

def main():

    print("=== SDR Алгоритм символьной синхронизации QAM ===")
    sdr = standart_settings("ip:192.168.2.1", 1e6, 10e3)

    print("\nВыберите схему модуляции:")
    print("1: QAM16")
    print("2: QAM64")
    choice = input("Введите номер схемы модуляции (по умолчанию 1): ") or "1"
    if choice == "1":
        mod_function = QAM16
        bits_per_symbol = 4
    elif choice == "2":
        mod_function = QAM64
        bits_per_symbol = 6
    else:
        print("[WARN] Неверный выбор, используется QAM16.")
        mod_function = QAM16
        bits_per_symbol = 4

    # === Генерация битовой последовательности ===
    bit_length = 2400
    bit_msg = randomDataGenerator(bit_length)
    valid_length = (len(bit_msg) // bits_per_symbol) * bits_per_symbol
    bit_msg = bit_msg[:valid_length]
    print(f"[INFO] Битовая последовательность длиной {len(bit_msg)} (округлена до кратности {bits_per_symbol}).")
    print("[DEBUG] Первые 20 бит:", bit_msg[:20])

    # === Формирование QAM символов ===
    qam_signal = mod_function(bit_msg)
    print(f"[INFO] Сформировано {len(qam_signal)} QAM символов.")

    signalRepeat = np.repeat(qam_signal, 10) * (2**14)
    print(f"[INFO] Длина передаваемого сигнала: {len(signalRepeat)} выборок.")

    # === TX/RX ===
    tx_signal(sdr, 2300e6, 0, signalRepeat)
    rx_sig = rx_signal(sdr, 2300e6, 20, 5)
    rx_sig = rx_sig / np.max(np.abs(rx_sig))
    print(f"[INFO] Полученный сигнал нормализован. Длина сигнала: {len(rx_sig)} выборок.")

    plot_constellation(rx_sig, title="Received Signal")

    # === Графики функции Gardner_TED ===
    error_before = gardner_ted_error(rx_sig, nsp=10)
    gardner_indices, error_after = gardner_ted(rx_sig, nsp=10, alpha=0.05)
    rx_after_ted = rx_sig[gardner_indices]
    print(f"[INFO] После временной синхронизации получено {len(rx_after_ted)} символов.")
    
    # Объединяем графики на одной фигуре
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(error_before, label="Error(n) ДО Gardner TED", color="blue")
    axs[0].set_title("График ошибки до символьной синхронизации")
    axs[0].set_ylabel("Error")
    axs[0].set_ylim([-2, 2])
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot(error_after, label="Error(n) ПОСЛЕ Gardner TED", color="green")
    axs[1].set_title("График ошибки после символьной синхронизации")
    axs[1].set_xlabel("n (symbol index)")
    axs[1].set_ylabel("Error")
    axs[1].set_ylim([-2, 2])
    axs[1].grid(True)
    axs[1].legend()
    fig.tight_layout()

    plot_constellation(rx_after_ted, title="After Gardner TED")

    # === Автокорреляция ===
    rx_after_freqcorr, freq_error = autocorr_freq_correct(rx_after_ted, n_lag=1)
    print(f"[INFO] После частотной коррекции (автокорреляция) частотная ошибка = {freq_error:.6e} рад/отсчёт")

    # === График фазовой синхронизации созвездия ===
    constellation = get_constellation(bits_per_symbol)
    rx_after_pll = PLL(rx_after_freqcorr, constellation, mu=0.05)
    plot_constellation(rx_after_pll, title="After PLL")

    plt.show()
    print("[INFO] Программа завершена.")

if __name__ == '__main__':
    main()