import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import adi

# === Функция задания параметров Adalm Pluto SDR ===
def standart_settings(Pluto_IP="192.168.2.1", sample_rate=1e6, buffer_size=1e3, gain_mode="manual"):
    sdr = adi.Pluto(Pluto_IP)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_buffer_size = int(buffer_size)
    sdr.gain_control_mode_chan0 = gain_mode
    print(f"[INFO] SDR настроен: sample_rate={sample_rate}, buffer_size={buffer_size}, gain_mode={gain_mode}")
    return sdr

# === Функция отрисовки графиков ===
def plot_constellation(symbols, title="Constellation Diagram", ax=None):
    if ax is None:
        fig, ax_new = plt.subplots(figsize=(6, 6))
    else:
        ax_new = ax
        ax_new.clear()

    ax_new.scatter(np.real(symbols), np.imag(symbols), color='blue', s=50, marker='o', edgecolors='k', alpha=0.8)
    ax_new.set_title(title)
    ax_new.set_xlabel("In-phase")
    ax_new.set_ylabel("Quadrature")
    ax_new.axhline(0, color='red', linewidth=1, linestyle='--')
    ax_new.axvline(0, color='red', linewidth=1, linestyle='--')
    ax_new.grid(True)
    ax_new.set_aspect('equal', 'box')
    max_val = np.max(np.abs(symbols)) * 1.2 if len(symbols) > 0 else 1.0
    ax_new.set_xlim(-max_val, max_val)
    ax_new.set_ylim(-max_val, max_val)
    if ax is None:
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

# === PLL-функция с возвратом ошибок ===
def PLL(signal, constellation, mu=0.05, verbose=True):
    output = np.zeros_like(signal, dtype=np.complex128)
    theta = 0.0
    phase_errors = np.zeros(len(signal), dtype=float)
    for n, x in enumerate(signal):
        x_rot = x * np.exp(-1j * theta)
        idx = np.argmin(np.abs(constellation - x_rot))
        decision = constellation[idx]
        err = np.angle(x_rot * np.conj(decision))
        phase_errors[n] = err
        theta += mu * err
        output[n] = x_rot
    if verbose:
        print(f"[INFO] PLL завершён. финальное theta={theta:.3f}, mean error={np.mean(phase_errors):.3e}")
    return output, phase_errors

# === Гарднер TED с управляемым накопителем (NCO) ===
def gardner_ted_synchronizer(sig, nsp=10, Kp=0.01, Ki=0.0001, alpha=0.1):
    N = len(sig)
    N_out_max = int(N / nsp * 1.5)
    output = np.zeros(N_out_max, dtype=sig.dtype)
    timing_errors = np.zeros(N_out_max)
    mu = 0.0
    v_p, v_i = 0.0, 0.0
    strobe = 0
    count = 0
    ema_error = 0.0

    while strobe < N - 2 - int(nsp):
        if count >= N_out_max:
            break

        # Линейная интерполяция - sample at strobe + mu
        y_k = sig[strobe] * (1 - mu) + sig[strobe + 1] * mu
        idx_mid = strobe - nsp//2
        if idx_mid < 0:
            strobe += int(nsp)
            continue
        y_mid = sig[idx_mid] * (1 - mu) + sig[idx_mid + 1] * mu
        idx_prev = strobe - nsp
        if idx_prev < 0:
            strobe += int(nsp)
            continue
        y_prev = sig[idx_prev] * (1 - mu) + sig[idx_prev + 1] * mu

        # Gardner error
        error = np.real((y_k - y_prev) * np.conj(y_mid))
        # PI-регулятор
        v_p = Kp * error
        v_i += Ki * error
        correction = v_p + v_i

        # Advance NCO
        total_advance = nsp - correction
        new_mu = (mu + total_advance)
        strobe += int(np.floor(new_mu))
        mu = new_mu % 1.0

        output[count] = y_k
        ema_error = alpha * error + (1 - alpha) * ema_error if count > 0 else error
        timing_errors[count] = ema_error
        count += 1

    print(f"[INFO] Gardner TED завершён. Получено {count} символов.")
    return output[:count], timing_errors[:count]

# === Gardner TED ошибка до синхронизации (визуализация) ===
def gardner_ted_error_calc(signal, nsp=10):
    num_potential_symbols = (len(signal) - nsp) // nsp
    errors = np.zeros(num_potential_symbols)
    count = 0
    for i in range(nsp // 2, len(signal) - nsp // 2 -1, nsp):
        if i >= nsp // 2 and i < len(signal) - nsp // 2 :
            s_early = signal[i - nsp // 2]
            s_mid   = signal[i]
            s_late  = signal[i + nsp // 2]
            err = np.real((s_late - s_early) * np.conj(s_mid))
            if count < num_potential_symbols:
                 errors[count] = err
                 count += 1
        if count >= num_potential_symbols:
            break
    print(f"[INFO] Gardner TED ошибка БЕЗ синхронизации рассчитана для {count} точек.")
    return errors[:count]

# --- Функция для частотной коррекции через автокорреляцию ---
def autocorr_freq_correct(signal, n_lag=1):
    x = signal
    if len(x) <= n_lag:
        print("[WARN] Сигнал слишком короткий для автокорреляционной коррекции.")
        return x, 0.0
    auto_corr_lagged = x[n_lag:] * np.conj(x[:-n_lag])
    mean_phase_offset = np.angle(np.mean(auto_corr_lagged))
    freq_error = mean_phase_offset / n_lag
    print(f"[INFO] Оценена частотная ошибка: {freq_error:.6f} рад/отсч.")
    n = np.arange(len(x))
    corrected = x * np.exp(-1j * freq_error * n)
    return corrected, freq_error

# === ГЕНЕРАЦИЯ БИТ ===
def randomDataGenerator(size):
    bits = np.random.randint(0, 2, size)
    print(f"[INFO] Сгенерировано {len(bits)} бит.")
    return bits

# === QAM16 ===
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
        raise Exception("QAM16: Длина битовой последовательности должна быть кратна 4.")
    symbols = np.array([qam16_table[tuple(bit_mass[i:i+4])] for i in range(0, len(bit_mass), 4)], dtype=np.complex128)
    print(f"[INFO] QAM16: Сформировано {len(symbols)} символов.")
    return symbols

# === QAM64 ===
def QAM64(bit_mass):
    gray_map_3bit = {
        (0,0,0): -7, (0,0,1): -5, (0,1,1): -3, (0,1,0): -1,
        (1,1,0):  1, (1,1,1):  3, (1,0,1):  5, (1,0,0):  7
    }
    if len(bit_mass) % 6 != 0:
        raise Exception("QAM64: Длина битовой последовательности должна быть кратна 6.")
    num_symbols = len(bit_mass) // 6
    symbols = np.zeros(num_symbols, dtype=np.complex128)
    for i in range(num_symbols):
        bits_real = tuple(bit_mass[i*6 : i*6+3])
        bits_imag = tuple(bit_mass[i*6+3 : i*6+6])
        real_val = gray_map_3bit[bits_real]
        imag_val = gray_map_3bit[bits_imag]
        symbols[i] = complex(real_val, imag_val)
    symbols /= np.sqrt(42)
    print(f"[INFO] QAM64: Сформировано {len(symbols)} символов.")
    return symbols

# === TX/RX functions ===
def tx_signal(sdr, tx_lo, gain_tx, data, tx_cycle=True):
    sdr.tx_lo = int(tx_lo)
    sdr.tx_hardwaregain_chan0 = gain_tx
    sdr.tx_cyclic_buffer = tx_cycle
    print(f"[INFO] Передача сигнала: tx_lo={tx_lo}, gain_tx={gain_tx}, cyclic={tx_cycle}")
    sdr.tx(data)

def rx_signal(sdr, rx_lo, gain_rx, num_buffers_to_capture):
    sdr.rx_lo = int(rx_lo)
    sdr.rx_hardwaregain_chan0 = gain_rx
    data = []
    for _ in range(num_buffers_to_capture):
        rx_samples = sdr.rx()
        data.append(rx_samples)
    print(f"[INFO] Принято {num_buffers_to_capture} блоков данных от SDR.")
    return np.concatenate(data)

def get_constellation(bits_per_symbol):
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
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        const_points = np.array([complex(r, i) for r in levels for i in levels])
        norm_factor = np.sqrt(42)
        return const_points / norm_factor
    else:
        raise ValueError("Unsupported bits_per_symbol for constellation!")

def close_sdr(sdr):
    try:
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
    except Exception as e:
        print(f"[INFO] Could not destroy SDR buffers: {e}")
        pass
    if sdr and hasattr(sdr, "close"):
        sdr.close()
    print("[INFO] SDR соединение закрыто и очищено.")

def calculate_evm(received_symbols, ideal_constellation_points):
    num_rx_symbols = len(received_symbols)
    evm_per_symbol = np.zeros(num_rx_symbols)
    ideal_symbols_for_rx = np.zeros(num_rx_symbols, dtype=np.complex128)
    error_vectors = np.zeros(num_rx_symbols, dtype=np.complex128)
    decision_indices = np.argmin(np.abs(received_symbols[:, np.newaxis] - ideal_constellation_points[np.newaxis, :]), axis=1)
    ideal_symbols_for_rx = ideal_constellation_points[decision_indices]
    error_vectors = received_symbols - ideal_symbols_for_rx
    abs_ideal_symbols = np.abs(ideal_symbols_for_rx)
    evm_per_symbol = np.where(abs_ideal_symbols > 1e-9, (np.abs(error_vectors) / abs_ideal_symbols) * 100, 100.0)
    mean_ideal_power = np.mean(np.abs(ideal_constellation_points)**2)
    if mean_ideal_power < 1e-9: mean_ideal_power = 1.0
    evm_rms = np.sqrt(np.mean(np.abs(error_vectors)**2) / mean_ideal_power) * 100
    evm_db = -np.inf
    if evm_rms > 1e-9:
        evm_db = 20 * np.log10(evm_rms / 100.0)
    return evm_per_symbol, evm_rms, evm_db, ideal_symbols_for_rx, error_vectors

# ===========================================================
#                      MAIN
# ===========================================================

def main():
    sdr = None
    try:
        print("=== SDR Алгоритм символьной синхронизации QAM ===")
        # --- Настройки SDR и модуляции ---
        pluto_ip = "ip:192.168.2.1"
        sample_rate = 1e6
        buffer_size_sdr = int(2**14)
        tx_lo_freq = 2100e6
        rx_lo_freq = 2100e6
        tx_gain = -10
        rx_gain = 20
        num_buffers_to_receive = 10
        bit_length = 4800
        samples_per_symbol_tx = 10

        sdr = standart_settings(pluto_ip, sample_rate, buffer_size_sdr, gain_mode="manual")
        sdr.rx_enabled_channels = [0]

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

        # === Генерация бит ===
        bit_msg_raw = randomDataGenerator(bit_length)
        valid_length = (len(bit_msg_raw) // bits_per_symbol) * bits_per_symbol
        bit_msg = bit_msg_raw[:valid_length]
        print(f"[INFO] Битовая последовательность скорректирована до длины {len(bit_msg)} (кратна {bits_per_symbol}).")

        # === QAM-сигнал ===
        qam_symbols_tx = mod_function(bit_msg)
        print(f"[INFO] Сформировано {len(qam_symbols_tx)} QAM символов для передачи.")

        signal_to_transmit = np.repeat(qam_symbols_tx, samples_per_symbol_tx) * (2**14)
        print(f"[INFO] Длина передаваемого сигнала: {len(signal_to_transmit)} выборок (отсчетов на символ: {samples_per_symbol_tx}).")

        # === TX/RX ===
        tx_signal(sdr, tx_lo_freq, tx_gain, signal_to_transmit, tx_cycle=True)
        rx_sig_raw = rx_signal(sdr, rx_lo_freq, rx_gain, num_buffers_to_receive)
        sdr.tx_destroy_buffer()
        print(f"[INFO] Принято {len(rx_sig_raw)} сырых выборок от SDR.")

        plot_constellation(rx_sig_raw, title="Received Signal")

        # RMS-нормализация
        rms_val = np.sqrt(np.mean(np.abs(rx_sig_raw)**2))
        if rms_val < 1e-6:
            print("[WARN] RMS принятого сигнала очень мал.")
            rx_sig_normalized = rx_sig_raw
        else:
            rx_sig_normalized = rx_sig_raw / rms_val
        print(f"[INFO] Полученный сигнал нормализован по RMS. Длина сигнала: {len(rx_sig_normalized)} выборок.")

        # Gardner TED ошибка до синхронизации (визуализация)
        error_before_sync = gardner_ted_error_calc(rx_sig_normalized, nsp=samples_per_symbol_tx)

        # === Символьная синхронизация Gardner TED ===
        gardner_kp = 0.01
        gardner_ki = 0.0001
        rx_after_gardner, ted_errors_during_sync = gardner_ted_synchronizer(
            rx_sig_normalized,
            nsp=samples_per_symbol_tx,
            Kp=gardner_kp,
            Ki=gardner_ki,
            alpha=0.1
        )
        print(f"[INFO] После символьной синхронизации Gardner TED получено {len(rx_after_gardner)} символов.")
        if len(rx_after_gardner) == 0:
            print("[ERROR] Gardner TED не вернул символы. Stop.")
            return

        plot_constellation(rx_after_gardner, title="After Gardner TED")

        # Графики ошибок Gardner TED
        fig_ted, axs_ted = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        axs_ted[0].plot(error_before_sync, label="TED Error BEFORE Sync", color="blue", markersize=2)
        axs_ted[0].set_title("Ошибка Gardner TED до символьной синхронизации")
        axs_ted[0].set_ylabel("Error")
        axs_ted[0].grid(True)
        axs_ted[0].legend()
        axs_ted[1].plot(ted_errors_during_sync, label="TED Error DURING Sync", color="green", markersize=2)
        axs_ted[1].set_title("Ошибка Gardner TED в процессе символьной синхронизации")
        axs_ted[1].set_xlabel("Symbol Index")
        axs_ted[1].set_ylabel("Error (smoothed)")
        axs_ted[1].grid(True)
        axs_ted[1].legend()
        fig_ted.tight_layout()

        # === Частотная коррекция через автокорреляцию ===
        rx_after_freqcorr, freq_error_rad_per_symbol = autocorr_freq_correct(rx_after_gardner, n_lag=1)
        print(f"[INFO] После частотной коррекции (автокорреляция) остаточная ошибка = {freq_error_rad_per_symbol:.6e} рад/символ")

        # PLL
        constellation_ref = get_constellation(bits_per_symbol)
        pll_mu = 0.1
        rx_after_pll, pll_phase_errors = PLL(rx_after_freqcorr, constellation_ref, mu=pll_mu)
        plot_constellation(rx_after_pll, title="After PLL Phase Correction")

        # EVM Analysis
        transient_skip = min(len(rx_after_pll) // 10, 50)
        if len(rx_after_pll) > transient_skip:
            symbols_for_evm = rx_after_pll[transient_skip:]
            evm_per_symbol, evm_rms, evm_db, ideal_sym_map, err_vec = calculate_evm(symbols_for_evm, constellation_ref)
            print(f"[INFO] EVM: RMS = {evm_rms:.2f}% ({evm_db:.2f} dB) для {len(symbols_for_evm)} символов")
            # График EVM vs Symbol
            plt.figure(figsize=(10, 4))
            plt.plot(evm_per_symbol, markersize=3, alpha=0.6)
            plt.title(f'EVM vs Symbol Index (RMS: {evm_rms:.2f}%)')
            plt.xlabel('Symbol Index (after transient skip)')
            plt.ylabel('EVM (%)')
            plt.grid(True)
            plt.ylim(0, max(50, np.max(evm_per_symbol)*1.1 if len(evm_per_symbol)>0 else 50) )
            plt.tight_layout()
            # График спектра ошибки
            if len(err_vec) > 1:
                fft_error = np.fft.fftshift(np.fft.fft(err_vec))
                fft_ideal = np.fft.fftshift(np.fft.fft(ideal_sym_map))
                freq_axis = np.fft.fftshift(np.fft.fftfreq(len(err_vec)))
                plt.figure(figsize=(10, 4))
                plt.plot(freq_axis, 20*np.log10(np.abs(fft_ideal)), label='Mapped Ideal Symbols Spectrum')
                plt.plot(freq_axis, 20*np.log10(np.abs(fft_error)), label='Error Vector Spectrum')
                plt.title('Frequency Domain: Mapped Ideal vs Error Vector')
                plt.xlabel('Normalized Frequency (cycles/symbol)')
                plt.ylabel('Magnitude (dB)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
        else:
            print("[INFO] Недостаточно символов для анализа EVM после пропуска переходного процесса.")

    except Exception as e:
        print(f"[ERROR] Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sdr:
            close_sdr(sdr)
        print("[INFO] Программа завершена.")

    plt.show()

if __name__ == '__main__':
    main()