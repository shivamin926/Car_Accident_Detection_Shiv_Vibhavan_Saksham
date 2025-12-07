document.getElementById("fileInput").addEventListener("change", function (event) {
            const file = event.target.files[0];
            if (!file) return;

            const url = URL.createObjectURL(file);

            const img = document.getElementById("preview");
            img.src = url;
            img.style.display = "block";
        });

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            if (fileInput.files.length === 0) {
                alert("Please select an image first!");
                return;
            }

            const file = fileInput.files[0];
            const binary = await file.arrayBuffer();

            const response = await fetch("http://127.0.0.1:8000/predict-image", {
                method: "POST",
                headers: {
                    "Content-Type": "application/octet-stream"
                },
                body: binary
            });

            const json = await response.json();

            const resultEl = document.getElementById('result');
            if (json.prediction === 1){
                resultEl.textContent = "The car is NOT damaged.";
                resultEl.className = "not-damaged";
            }
            else {
                resultEl.textContent = "The car IS damaged";
                resultEl.className = "damaged";
            }

        }