/**
 * Video Scene App - Static design interaction
 * Minimal JS for Lucide icons and basic UX
 */
document.addEventListener('DOMContentLoaded', function () {
  // Initialize Lucide icons
  if (typeof lucide !== 'undefined') {
    lucide.createIcons();
  }

  // Chip filter toggle
  const chips = document.querySelectorAll('.chip');
  chips.forEach(function (chip) {
    chip.addEventListener('click', function () {
      chips.forEach(function (c) { c.classList.remove('active'); });
      chip.classList.add('active');
    });
  });

  // Tab bar navigation (visual only - static demo)
  const tabItems = document.querySelectorAll('.tab-item');
  tabItems.forEach(function (tab) {
    tab.addEventListener('click', function (e) {
      e.preventDefault();
      tabItems.forEach(function (t) { t.classList.remove('active'); });
      tab.classList.add('active');
    });
  });

  // Help accordion - toggle expanded item
  const helpItems = document.querySelectorAll('.help-item');
  helpItems.forEach(function (item) {
    item.addEventListener('click', function () {
      const expanded = document.querySelector('.help-item.expanded');
      if (expanded && expanded !== item) {
        expanded.classList.remove('expanded');
      }
      if (item.classList.contains('expanded')) {
        item.classList.remove('expanded');
      } else if (item.querySelector('.help-item-content')) {
        item.classList.add('expanded');
      }
    });
  });
});
