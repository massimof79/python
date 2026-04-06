document.getElementById("formCasa").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        rm: document.getElementById("rm").value,
        lstat: document.getElementById("lstat").value,
        ptratio: document.getElementById("ptratio").value,
        tax: document.getElementById("tax").value,
        age: document.getElementById("age").value
    };

    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("risultato").innerText =
        "Prezzo stimato dell'abitazione: " + result.predicted_price + " (migliaia di dollari)";
});
