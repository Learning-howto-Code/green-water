document.addEventListener("keydown", function(event) {
  const key = event.key.toLowerCase();
  const allowed = ["w", "a", "s", "d", "x"];
  if (allowed.includes(key)) {
    fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: "key=" + key
    });
  }
});
