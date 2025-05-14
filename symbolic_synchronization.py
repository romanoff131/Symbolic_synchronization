import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import adi
import random

def standart_settings(Pluto_IP="192.168.3.1", sample_rate=1e6, buffer_size=1e3, gain_mode="manual"):
    sdr = adi.Pluto(Pluto_IP)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.gain_control_mode_chan0 = gain_mode
    print(f"[INFO] SDR настроен: sample_rate={sample_rate}, buffer_size={buffer_size}, gain_mode={gain_mode}")
    return sdr

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

def PLL(conv):
    mu = 0.1
    theta = 0
    phase_error = np.zeros(len(conv))
    output_signal = np.zeros(len(conv), dtype=np.complex128)
    for n in range(len(conv)):
        theta_hat = np.angle(conv[n])
        phase_error[n] = theta_hat - theta
        output_signal[n] = conv[n] * np.exp(-1j * theta)
        theta = theta + mu * phase_error[n]
    print(f"[INFO] PLL завершён. Итоговое значение theta = {theta:.3f}")
    return output_signal

def gardner_ted(signal, nsp=10):
    """
    Возвращает индексы и вектор ошибки -- для последующего графика!
    """
    error = np.zeros(len(signal) // nsp)
    timing_offsets = []
    index_sync = []
    offset = 0
    for i in range(nsp, len(signal) - nsp, nsp):
        s_early = signal[i - nsp//2]
        s_mid = signal[i]
        s_late = signal[i + nsp//2]
        err = (s_late - s_early) * np.conj(s_mid)
        error[i // nsp] = np.real(err)

        offset += np.real(err) * 0.01  # условная динамика
        timing_offsets.append(offset)
        index_sync.append(i)

    # RENDER improved error plot
    plt.figure(figsize=(8, 4))
    plt.plot(error, label="Gardner Timing Error", lw=1)
    plt.title("Gardner TED Timing Error per Symbol")
    plt.xlabel("Symbol Index (not sample!)")
    plt.ylabel("Timing Error")
    plt.axhline(0, color='red', linestyle='--', label='Zero error')
    # Добавим скользящее среднее
    window = 15
    if len(error) > window:
        moving_avg = np.convolve(error, np.ones(window)/window, mode='valid')
        plt.plot(range(window//2,window//2+len(moving_avg)), moving_avg, 'g', lw=2, label='Moving Avg')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Вывод динамики смещения момента
    plt.figure(figsize=(8, 4))
    plt.plot(timing_offsets, label="Timing Offset (Dynamic)")
    plt.title("Timing Offset Dynamics (Gardner TED)")
    plt.xlabel("Symbol index")
    plt.ylabel("Timing Offset [a.u.]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

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

def randomDataGenerator(size):
    bits = [random.randint(0, 1) for _ in range(size)]
    print(f"[INFO] Сгенерировано {len(bits)} бит.")
    return bits

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

def QAM16_demod(rx_symbols):
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
    constellation = np.array(list(qam16_table.values()))
    bits_list = list(qam16_table.keys())
    demod_bits = []
    for symbol in rx_symbols:
        dist = np.abs(symbol - constellation)
        idx = np.argmin(dist)
        demod_bits.extend(bits_list[idx])
    print(f"[INFO] QAM16 демодулировано {len(rx_symbols)} символов в {len(demod_bits)} бит.")
    return np.array(demod_bits, dtype=np.uint8)

def QAM64_demod(rx_symbols):
    ampl = 2**14
    if np.max(np.abs(rx_symbols)) > 8:
        rx_symbols = rx_symbols / ampl
    rx_symbols = rx_symbols * np.sqrt(42)
    decision_levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    val_to_bits = {v: [int(x) for x in format(((v + 7) // 2), '03b')] for v in decision_levels}
    result_bits = []
    for symbol in rx_symbols:
        i = decision_levels[np.argmin(np.abs(np.real(symbol) - decision_levels))]
        q = decision_levels[np.argmin(np.abs(np.imag(symbol) - decision_levels))]
        result_bits.extend(val_to_bits[i] + val_to_bits[q])
    print(f"[INFO] QAM64 демодулировано {len(rx_symbols)} символов в {len(result_bits)} бит.")
    return np.array(result_bits, dtype=np.uint8)

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

def main():
    print("=== SDR Алгоритм символьной синхронизации QAM ===")
    sdr = standart_settings("ip:192.168.3.1", 1e6, 10e3)

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

    bit_length = 2400
    bit_msg = randomDataGenerator(bit_length)
    valid_length = (len(bit_msg) // bits_per_symbol) * bits_per_symbol
    bit_msg = bit_msg[:valid_length]
    print(f"[INFO] Битовая последовательность длиной {len(bit_msg)} (округлена до кратности {bits_per_symbol}).")
    print("[DEBUG] Первые 20 бит:", bit_msg[:20])

    qam_signal = mod_function(bit_msg)
    print(f"[INFO] Сформировано {len(qam_signal)} QAM символов.")

    signalRepeat = np.repeat(qam_signal, 10) * (2**14)
    print(f"[INFO] Длина передаваемого сигнала: {len(signalRepeat)} выборок.")

    tx_signal(sdr, 2300e6, 0, signalRepeat)
    rx_sig = rx_signal(sdr, 2300e6, 20, 5)
    rx_sig = rx_sig / np.max(np.abs(rx_sig))
    print(f"[INFO] Полученный сигнал нормализован. Длина сигнала: {len(rx_sig)} выборок.")

    # === Q1: Вернуть графики по Gardner ===
    plot_constellation(rx_sig, title="Received Signal")
    gardner_indices, gardner_error = gardner_ted(rx_sig, nsp=10)
    rx_after_ted = rx_sig[gardner_indices]
    print(f"[INFO] После временной синхронизации получено {len(rx_after_ted)} символов.")
    plot_constellation(rx_after_ted, title="After Gardner TED")

    # === Q2: Только автокорреляция без отображения отдельного графика ===
    rx_after_freqcorr, freq_error = autocorr_freq_correct(rx_after_ted, n_lag=1)
    print(f"[INFO] После частотной коррекции (автокорреляция) частотная ошибка = {freq_error:.6e} рад/отсчёт")
    # plot_constellation(rx_after_freqcorr, title="After Frequency Correction") # отключено по просьбе

    rx_after_pll = PLL(rx_after_freqcorr)
    plot_constellation(rx_after_pll, title="After PLL")

    if bits_per_symbol == 4:
        demod_bits = QAM16_demod(rx_after_pll)
    elif bits_per_symbol == 6:
        demod_bits = QAM64_demod(rx_after_pll)
    else:
        demod_bits = None
    print(f"[INFO] Демодулированные биты (первые 20): {demod_bits[:20]}")

    plt.show()
    print("[INFO] Программа завершена.")

if __name__ == '__main__':
    main()