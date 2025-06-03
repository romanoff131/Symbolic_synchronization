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

# === Фазовая синхронизация Decision-Directed PLL ===
def dd_pll_qam(received_syms, constellation, loop_bw=0.01, damping=0.707, verbose=True):
    
    N = len(received_syms)
    out_syms = np.zeros(N, dtype=np.complex128)
    phase_errs = np.zeros(N)
    est_phase = 0.0

    zeta = damping
    wn = loop_bw
    denom = 1 + 2 * zeta * wn + wn**2
    alpha = (4 * zeta * wn) / denom
    beta = (4 * wn**2) / denom
    int_err = 0.0
    theta = est_phase

    for n in range(N):
        rx_rot = received_syms[n] * np.exp(-1j * theta)
        idx = np.argmin(np.abs(rx_rot - constellation))
        decision = constellation[idx]
        err = np.angle(rx_rot * np.conj(decision))
        theta += alpha * err + int_err
        int_err += beta * err

        out_syms[n] = rx_rot
        phase_errs[n] = err

    if verbose:
        print(f"Средняя ошибка фазы: {np.mean(np.abs(phase_errs))}")
    return out_syms, phase_errs

# === Гарднер TED ===
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

# === Gardner TED ошибка до синхронизации ===
def gardner_ted_error_calc(signal, nsp=10):
    num_potential_symbols = (len(signal) - nsp) // nsp
    errors = np.zeros(num_potential_symbols)
    count = 0
    for i in range(nsp // 2, len(signal) - nsp // 2 -1, nsp):
        if i >= nsp // 2 and i < len(signal) - nsp // 2:
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

# === генерация бит ===
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

def advanced_amplitude_affine_correction(symbols_rx, ideal_constellation):
    if len(symbols_rx) == 0:
        return symbols_rx, (1.0+0j, 0.0+0j, 0.0+0j)
    idx_nearest = np.argmin(np.abs(symbols_rx[:, None] - ideal_constellation[None, :]), axis=1)
    nearest_const = ideal_constellation[idx_nearest]
    X = np.vstack([symbols_rx, np.conj(symbols_rx), np.ones_like(symbols_rx)]).T
    Y = nearest_const
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    a, b, c = coeffs
    corrected = a * symbols_rx + b * np.conj(symbols_rx) + c
    return corrected, (a, b, c)

# === Комппенсация глобального фазового сдвига ===
def global_phase_compensation(received_syms, reference_syms):
    if len(received_syms) == 0 or len(reference_syms) == 0:
        return received_syms, 0.0
    angle_offset = np.angle(np.sum(reference_syms * np.conj(received_syms)))
    corrected_syms = received_syms * np.exp(1j * angle_offset)
    return corrected_syms, angle_offset

def main():
    sdr = None
    try:
        print("=== SDR Алгоритм символьной синхронизации QAM ===")
        pluto_ip = "ip:192.168.3.1"
        sample_rate = 1e6
        buffer_size_sdr = int(2**14)
        tx_lo_freq = 2100e6
        rx_lo_freq = 2100e6
        tx_gain = 0
        rx_gain = 0
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

        bit_msg_raw = randomDataGenerator(bit_length)
        valid_length = (len(bit_msg_raw) // bits_per_symbol) * bits_per_symbol
        bit_msg = bit_msg_raw[:valid_length]
        print(f"[INFO] Битовая последовательность скорректирована до длины {len(bit_msg)} (кратна {bits_per_symbol}).")

        qam_symbols_tx = mod_function(bit_msg)
        print(f"[INFO] Сформировано {len(qam_symbols_tx)} QAM символов для передачи.")

        signal_to_transmit = np.repeat(qam_symbols_tx, samples_per_symbol_tx) * (2**14)
        print(f"[INFO] Длина передаваемого сигнала: {len(signal_to_transmit)} выборок (отсчетов на символ: {samples_per_symbol_tx}).")

        # TX/RX
        tx_signal(sdr, tx_lo_freq, tx_gain, signal_to_transmit, tx_cycle=True)
        rx_sig_raw = rx_signal(sdr, rx_lo_freq, rx_gain, num_buffers_to_receive)
        sdr.tx_destroy_buffer()
        print(f"[INFO] Принято {len(rx_sig_raw)} сырых выборок от SDR.")

        plot_constellation(rx_sig_raw, title="Received Signal")

        rms_val = np.sqrt(np.mean(np.abs(rx_sig_raw)**2))
        if rms_val < 1e-6:
            print("[WARN] RMS принятого сигнала очень мал.")
            rx_sig_normalized = rx_sig_raw
        else:
            rx_sig_normalized = rx_sig_raw / rms_val
        print(f"[INFO] Полученный сигнал нормализован по RMS. Длина сигнала: {len(rx_sig_normalized)} выборок.")

        error_before_sync = gardner_ted_error_calc(rx_sig_normalized, nsp=samples_per_symbol_tx)

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

        # === DD-PLL ===
        constellation_ref = get_constellation(bits_per_symbol)
        pll_loop_bw = 0.02
        pll_damping = 0.7
        rx_after_pll, pll_phase_errors = dd_pll_qam(
            rx_after_gardner,
            constellation_ref,
            loop_bw=pll_loop_bw,
            damping=pll_damping,
            verbose=True
        )   
        plot_constellation(rx_after_pll, title="After QAM DD-PLL Phase Tracker")

        # Аффинная коррекция амплитуды
        corrected_rx_after_pll, abc_coeffs = advanced_amplitude_affine_correction(rx_after_pll, constellation_ref)
        a, b, c = abc_coeffs
        print(f"[INFO] Амплитудная коррекция (расширенная):", end=' ')
        print(f'a={a:.4f}, b={b:.4f}, c={c:.4f}')

        # EVM Analysis
        transient_skip = min(len(corrected_rx_after_pll) // 10, 50)
        if len(corrected_rx_after_pll) > transient_skip:
            symbols_for_evm = corrected_rx_after_pll[transient_skip:]
            evm_per_symbol, evm_rms, evm_db, ideal_sym_map, err_vec = calculate_evm(symbols_for_evm, constellation_ref)

            symbols_for_evm_phasecorr, phase_correction_angle = global_phase_compensation(symbols_for_evm, ideal_sym_map)
            print(f"[INFO] Фазовый сдвиг (глобальная компенсация): {np.degrees(phase_correction_angle):.2f} град.")

            evm_per_symbol2, evm_rms2, evm_db2, ideal_sym_map2, err_vec2 = calculate_evm(symbols_for_evm_phasecorr, constellation_ref)
            evm_peak2 = np.max(evm_per_symbol2)
            evm_95_2 = np.percentile(evm_per_symbol2, 95)

            print(f"[INFO] EVM ПОСЛЕ фазовой компенсации: RMS = {evm_rms2:.2f}% ({evm_db2:.2f} dB) для {len(symbols_for_evm_phasecorr)} символов")
            print(f"[INFO] Пиковое значение EVM = {evm_peak2:.2f}%")
            print(f"[INFO] 95-й процентиль EVM = {evm_95_2:.2f}%")

            plt.figure(figsize=(7, 7))
            plt.scatter(np.real(ideal_sym_map2), np.imag(ideal_sym_map2), c="limegreen", s=16, alpha=0.85, label="Ideal (Mapped)")
            plt.scatter(np.real(symbols_for_evm_phasecorr), np.imag(symbols_for_evm_phasecorr), c="royalblue", s=16, alpha=0.4, label="Received (Measured, phase corr)")
            step = max(1, len(symbols_for_evm)//150)
            for idx in range(0, len(symbols_for_evm_phasecorr), step):
                plt.arrow(
                    np.real(ideal_sym_map2[idx]), np.imag(ideal_sym_map2[idx]),
                    np.real(err_vec2[idx]), np.imag(err_vec2[idx]),
                    color="crimson", width=0.002, head_width=0.05, alpha=0.7, length_includes_head=True
                )
            plt.title("EVM vectors")
            plt.xlabel("In-phase")
            plt.ylabel("Quadrature")
            plt.grid(True)
            plt.legend(loc="best")
            plt.axis("equal")
            plt.tight_layout()
            plt.figure(figsize=(10, 4))
            plt.plot(evm_per_symbol2, markersize=3, alpha=0.6)
            plt.title(f'EVM vs Symbol Index (RMS: {evm_rms2:.2f}%) — Phase Corrected')
            plt.xlabel('Symbol Index (after transient skip)')
            plt.ylabel('EVM (%)')
            plt.grid(True)
            plt.ylim(0, max(50, np.max(evm_per_symbol2)*1.1 if len(evm_per_symbol2)>0 else 50) )
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