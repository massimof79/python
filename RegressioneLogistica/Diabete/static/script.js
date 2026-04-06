document.getElementById("formDiabete").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        pregnancies: document.getElementById("pregnancies").value,
        glucose: document.getElementById("glucose").value,
        bloodpressure: document.getElementById("bloodpressure").value,
        skinthickness: document.getElementById("skinthickness").value,
        insulin: document.getElementById("insulin").value,
        bmi: document.getElementById("bmi").value,
        dpf: document.getElementById("dpf").value,
        age: document.getElementById("age").value
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("risultato").innerText =
        "Probabilità diabete: " + result.probability +
        " | Classe prevista: " + result.prediction;
});
