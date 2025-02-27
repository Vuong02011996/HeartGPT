import subprocess


def run_bxb(info_bxb: dict):

    report_line_file = info_bxb["report_line_file"]
    sd_file = info_bxb["sd_file"]
    report_standard_file = info_bxb["report_standard_file"]
    save_path = info_bxb["save_path"]
    filename = info_bxb["filename"]
    ref_ext = info_bxb["ref_ext"]

    cmd1 = f"bxb -r {filename} -a 'atr' '{ref_ext}' -L {report_line_file} {sd_file} -f '0'"
    cmd2 = f"bxb -r {filename} -a 'atr' '{ref_ext}' -S {report_standard_file} -f '0'"

    subprocess.run(cmd1, cwd=save_path, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd2, cwd=save_path, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # bxb -r 275 -a "atr" "ai" -S 275_report_standard.out  -f "0"
    # bxb -r 275 -a "atr" "ai" -L 275_report_line.out sd.out -f "0"


def run_sumstats(file_path: str):
    cmd = ["sumstats", file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        # print("Sumstats output:")
        # print(result.stdout)
        return result.stdout
    else:
        print("Error running sumstats:")
        print(result.stderr)
        return None

if __name__ == '__main__':
    pass