import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt


# Parametry obrazów twarzy
IMG_WIDTH, IMG_HEIGHT = 100, 100


class BiometricApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System biometryczny: twarz + odcisk")
        self.root.geometry("1000x750")

        self.face_train_dir = None
        self.fingerprint_train_dir = None
        self.face_test_img = None
        self.fingerprint_test_img = None

        self.pca_model = None
        self.X_pca = None
        self.face_file_names = []

        self.knn = None
        self.fingerprint_label_map = {}

        self._build_gui()

    # GUI

    def _build_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        base_frame = ttk.LabelFrame(main_frame, text="Bazy danych")
        base_frame.pack(fill="x", pady=5)

        ttk.Button(
            base_frame,
            text="1. Wybierz folder bazy TWARZY",
            command=self.choose_face_train_dir
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lbl_face_train = ttk.Label(base_frame, text="Nie wybrano folderu")
        self.lbl_face_train.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(
            base_frame,
            text="2. Wybierz folder bazy ODCISKÓW",
            command=self.choose_fingerprint_train_dir
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.lbl_fingerprint_train = ttk.Label(base_frame, text="Nie wybrano folderu")
        self.lbl_fingerprint_train.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        test_frame = ttk.LabelFrame(main_frame, text="Obrazy do identyfikacji")
        test_frame.pack(fill="x", pady=5)

        ttk.Button(
            test_frame,
            text="3. Wybierz obraz TWARZY",
            command=self.choose_face_test_image
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.lbl_face_test = ttk.Label(test_frame, text="Brak pliku")
        self.lbl_face_test.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(
            test_frame,
            text="4. Wybierz obraz ODCISKU",
            command=self.choose_fingerprint_test_image
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.lbl_fingerprint_test = ttk.Label(test_frame, text="Brak pliku")
        self.lbl_fingerprint_test.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill="x", pady=10)

        ttk.Button(
            action_frame,
            text="Analizuj bazy (PCA twarzy + LBP k-NN odcisków)",
            command=self.analyze_bases
        ).pack(side="left", padx=5)

        ttk.Button(
            action_frame,
            text="Uruchom algorytm poszukujący",
            command=self.run_search
        ).pack(side="left", padx=5)

        self.log_area = scrolledtext.ScrolledText(main_frame, width=120, height=25)
        self.log_area.pack(fill="both", expand=True, pady=5)

    def choose_face_train_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.face_train_dir = path
            self.lbl_face_train.config(text=path)

    def choose_fingerprint_train_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.fingerprint_train_dir = path
            self.lbl_fingerprint_train.config(text=path)

    def choose_face_test_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.jpg *.png *.jpeg *.bmp")]
        )
        if path:
            self.face_test_img = path
            self.lbl_face_test.config(text=os.path.basename(path))

    def choose_fingerprint_test_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Obrazy", "*.jpg *.png *.bmp *.tif *.tiff *.jpeg")]
        )
        if path:
            self.fingerprint_test_img = path
            self.lbl_fingerprint_test.config(text=os.path.basename(path))

    def analyze_bases(self):
        self.log_area.delete(1.0, tk.END)
        ok_face = self.run_pca_on_folder()
        ok_fingerprint = self.train_lbp_knn()

        if ok_face:
            self.log_area.insert(tk.END, "Analiza PCA bazy twarzy zakończona.\n")
        if ok_fingerprint:
            self.log_area.insert(tk.END, "Trenowanie LBP + kNN dla odcisków zakończone.\n")

        if ok_face and ok_fingerprint:
            messagebox.showinfo("Informacja", "Analiza obu baz zakończona pomyślnie.")
        elif ok_face or ok_fingerprint:
            messagebox.showwarning(
                "Uwaga",
                "Udało się przeanalizować tylko jedną z baz. Sprawdź logi."
            )

    # PCA dla twarzy

    def load_faces_from_folder(self, folder):
        X = []
        files = []
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                path = os.path.join(folder, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.log_area.insert(tk.END, f"Nie można wczytać obrazu: {path}\n")
                    continue
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                X.append(img.flatten())
                files.append(filename)
        return np.array(X), files

    def run_pca_on_folder(self):
        if not self.face_train_dir:
            messagebox.showwarning("Błąd", "Nie wybrano folderu bazy twarzy.")
            return False

        X, self.face_file_names = self.load_faces_from_folder(self.face_train_dir)
        if len(X) == 0:
            messagebox.showerror("Błąd", "Brak poprawnych obrazów twarzy w folderze.")
            return False

        n_components = min(10, X.shape[0], X.shape[1])
        self.pca_model = PCA(
            n_components=n_components,
            whiten=True,
            random_state=42
        )
        self.X_pca = self.pca_model.fit_transform(X)
        return True

    def recognize_face(self):
        if self.pca_model is None or self.X_pca is None:
            messagebox.showwarning(
                "Uwaga",
                "Najpierw wykonaj analizę PCA na bazie twarzy."
            )
            return

        if not self.face_test_img:
            messagebox.showwarning(
                "Uwaga",
                "Najpierw wybierz obraz twarzy do rozpoznania."
            )
            return

        img = cv2.imread(self.face_test_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror(
                "Błąd",
                "Nie można wczytać obrazu twarzy do rozpoznania."
            )
            return

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_flat = img.flatten().reshape(1, -1)
        img_pca = self.pca_model.transform(img_flat)

        distances = np.linalg.norm(self.X_pca - img_pca, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        threshold = 1.1 

        self.log_area.insert(tk.END, "\n>>> WYNIK KLASYFIKACJI TWARZY <<<\n")
        self.log_area.insert(
            tk.END,
            f"Najbardziej podobny obraz w bazie: {self.face_file_names[min_idx]}\n"
        )
        self.log_area.insert(tk.END, f"Dystans (różnica): {min_dist:.6f}\n")
        if min_dist < threshold:
            self.log_area.insert(
                tk.END,
                "Werdykt: Tożsamość POTWIERDZONA (duże podobieństwo).\n"
            )
        else:
            self.log_area.insert(
                tk.END,
                "Werdykt: Tożsamość NIEPEWNA (spora różnica w teksturze).\n"
            )

    # LBP + kNN dla odcisków

    def compute_lbp_hist(self, img_path):
        img = cv2.imread(img_path, 0)
        if img is None:
            return None
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        return hist.astype(np.float32)

    def train_lbp_knn(self):
        if not self.fingerprint_train_dir:
            messagebox.showwarning(
                "Błąd",
                "Nie wybrano folderu bazy odcisków."
            )
            return False

        train_data = []
        train_labels = []
        self.fingerprint_label_map = {}
        label_counter = 0

        try:
            valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            all_files = os.listdir(self.fingerprint_train_dir)
            files = [f for f in all_files if f.lower().endswith(valid_ext)]

            if not files:
                messagebox.showerror(
                    "Błąd",
                    "Nie znaleziono obrazów odcisków w folderze."
                )
                return False

            for filename in files:
                full_path = os.path.join(self.fingerprint_train_dir, filename)
                hist = self.compute_lbp_hist(full_path)
                if hist is not None:
                    train_data.append(hist)
                    train_labels.append(label_counter)
                    self.fingerprint_label_map[label_counter] = filename
                    label_counter += 1
                else:
                    self.log_area.insert(
                        tk.END,
                        f"Błąd odczytu pliku odcisku: {filename}\n"
                    )

            if not train_data:
                messagebox.showerror(
                    "Błąd",
                    "Nie udało się przetworzyć żadnego pliku odcisku."
                )
                return False

            train_matrix = np.array(train_data, dtype=np.float32)
            labels_matrix = np.array(train_labels, dtype=np.int32)

            self.knn = cv2.ml.KNearest_create()
            self.knn.train(train_matrix, cv2.ml.ROW_SAMPLE, labels_matrix)
            return True

        except Exception as e:
            messagebox.showerror("Błąd przetwarzania odcisków", str(e))
            return False

    def recognize_fingerprint(self):
        if self.knn is None:
            messagebox.showwarning(
                "Uwaga",
                "Najpierw wytrenuj model LBP + kNN dla odcisków (Analizuj bazy)."
            )
            return

        if not self.fingerprint_test_img:
            messagebox.showwarning(
                "Uwaga",
                "Najpierw wybierz obraz odcisku do rozpoznania."
            )
            return

        test_hist = self.compute_lbp_hist(self.fingerprint_test_img)
        if test_hist is None:
            messagebox.showerror(
                "Błąd",
                "Nie można wczytać obrazu odcisku do rozpoznania."
            )
            return

        test_sample = np.array([test_hist], dtype=np.float32)
        k = 1
        ret, results, neighbours, dist = self.knn.findNearest(test_sample, k)
        found_id = int(results[0][0])
        found_file = self.fingerprint_label_map.get(found_id, "Nieznany")
        distance = float(dist[0][0])

        self.log_area.insert(tk.END, "\n>>> WYNIK KLASYFIKACJI ODCISKU <<<\n")
        self.log_area.insert(
            tk.END,
            f"Najbardziej podobny obraz w bazie: {found_file}\n"
        )
        self.log_area.insert(tk.END, f"Dystans (różnica): {distance:.6f}\n")
        if distance < 0.05:
            self.log_area.insert(
                tk.END,
                "Werdykt: Tożsamość POTWIERDZONA (duże podobieństwo).\n"
            )
        else:
            self.log_area.insert(
                tk.END,
                "Werdykt: Tożsamość NIEPEWNA (spora różnica w teksturze).\n"
            )

    def run_search(self):
        self.log_area.insert(
            tk.END,
            "\n====== URUCHOMIENIE ALGORYTMU POSZUKUJĄCEGO ======\n"
        )
        self.recognize_face()
        self.recognize_fingerprint()
        self.log_area.insert(
            tk.END,
            "====== KONIEC URUCHOMIENIA ALGORYTMU ======\n"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = BiometricApp(root)
    root.mainloop()
