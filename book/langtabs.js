(function loadDeviconCSS() {
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href =
    "https://cdn.jsdelivr.net/gh/devicons/devicon@v2.16.0/devicon.min.css";
  document.head.appendChild(link);
})();

// Key for localStorage
const LANG_PREFERENCE_KEY = "mdbook-langtabs-preference";

document.addEventListener("DOMContentLoaded", function () {
  initLangTabs();

  // Listen for theme changes to re-style tabs
  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      if (
        mutation.attributeName === "class" &&
        mutation.target.nodeName === "HTML"
      ) {
        setTimeout(initLangTabs, 50);
      }
    });
  });

  observer.observe(document.documentElement, {
    attributes: true,
  });

  // Also handle theme changes when page hash changes (mdbook sometimes updates theme this way)
  window.addEventListener("hashchange", function () {
    setTimeout(initLangTabs, 100);
  });

  // Handle page navigation
  window.addEventListener("load", function () {
    setTimeout(initLangTabs, 100);
  });
});

function getSavedLanguage() {
  try {
    return localStorage.getItem(LANG_PREFERENCE_KEY);
  } catch (e) {
    return null;
  }
}

function saveLanguage(lang) {
  try {
    localStorage.setItem(LANG_PREFERENCE_KEY, lang);
  } catch (e) {
    // localStorage might be disabled
  }
}

function initLangTabs() {
  const langTabsContainers = document.querySelectorAll(".langtabs");

  langTabsContainers.forEach(function (container) {
    const tabButtons = container.querySelectorAll(".langtabs-tab");

    tabButtons.forEach(function (button) {
      button.removeEventListener("click", handleTabClick);
      button.addEventListener("click", handleTabClick);
    });
  });

  // Apply saved language preference or activate first tab
  const savedLang = getSavedLanguage();
  if (savedLang) {
    switchAllToLanguage(savedLang);
  } else {
    // If no saved preference, activate the first tab in each container
    langTabsContainers.forEach(function (container) {
      if (!container.querySelector(".langtabs-tab.active")) {
        const firstButton = container.querySelector(".langtabs-tab");
        if (firstButton) {
          const lang = firstButton.getAttribute("data-lang");
          activateLanguageInContainer(container, lang);
        }
      }
    });
  }
}

function handleTabClick() {
  const lang = this.getAttribute("data-lang");

  // Save the preference
  saveLanguage(lang);

  // Switch all code blocks to this language
  switchAllToLanguage(lang);
}

function switchAllToLanguage(lang) {
  const langTabsContainers = document.querySelectorAll(".langtabs");

  langTabsContainers.forEach(function (container) {
    activateLanguageInContainer(container, lang);
  });
}

function activateLanguageInContainer(container, lang) {
  const tabButtons = container.querySelectorAll(".langtabs-tab");
  const tabContents = container.querySelectorAll(".langtabs-code");

  // Check if this language exists in this container
  const targetButton = container.querySelector(
    `.langtabs-tab[data-lang="${lang}"]`,
  );
  const targetContent = container.querySelector(
    `.langtabs-code[data-lang="${lang}"]`,
  );

  if (!targetButton || !targetContent) {
    // Language not available in this container, fall back to first available
    const firstButton = container.querySelector(".langtabs-tab");
    if (firstButton) {
      lang = firstButton.getAttribute("data-lang");
    } else {
      return; // No tabs in this container
    }
  }

  // Deactivate all tabs and contents
  tabButtons.forEach(function (btn) {
    btn.classList.remove("active");
  });
  tabContents.forEach(function (content) {
    content.classList.remove("active");
  });

  // Activate the selected language
  const activeButton = container.querySelector(
    `.langtabs-tab[data-lang="${lang}"]`,
  );
  const activeContent = container.querySelector(
    `.langtabs-code[data-lang="${lang}"]`,
  );

  if (activeButton) {
    activeButton.classList.add("active");
  }
  if (activeContent) {
    activeContent.classList.add("active");
  }
}
