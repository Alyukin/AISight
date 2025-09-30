<script setup>
import DicomViewer from './DicomViewer.vue'
</script>

<template>
  <div class="app__section">   
    <header class="container inter__bold text-[clamp(20px,4vw,30px)] text-[#f0f0f0]">
      <h1 class="drop__shadow">AISight</h1>
    </header>
  
    <div class="container inter__bold text-[clamp(28px,6vw,60px)] mt-7 text-[#f0f0f0]">
      <h1 class="drop__shadow">Выявление патологий органов <br>грудной клетки</h1>
    </div>

    <div class="ct__window container">
        <DicomViewer />
    </div>
    
    <!-- Контейнер benefits -->
    <div class="benefits__container shadow-lg flex flex-wrap gap-13">
      <div class="benefits__item relative flex-1">
        <img src="/icon1.png" alt="Overlay Image" class="overlay-image" />
        <div class="benefits__tile">
          <h3 class="inter__semibold text-[22px] mb-4">Автоматический Анализ <br>КТ-Исследований</h3>
          <p class="text-[#ACACAC]">Определяйте наличие патологий, загрузив серии снимков пациента в 3D формате</p>
        </div>
      </div>
      <div class="benefits__item relative flex-1">
        <img src="/icon2.png" alt="Overlay Image" class="overlay-image" />
        <div class="benefits__tile">
          <h3 class="inter__semibold text-[22px] mb-4">Готовый DICOM пайплайн</h3>
          <p class="text-[#ACACAC]">Чистые, стандартизированные данные для обучения ИИ</p>
        </div>
      </div>
      <div class="benefits__item relative flex-1">
        <img src="/icon3.png" alt="Overlay Image" class="overlay-image" />
        <div class="benefits__tile">
          <h3 class="inter__semibold text-[22px] mb-4">Открытый исходный код</h3>
          <p class="text-[#ACACAC]">Работает на базе MONAI, открыт для сотрудничества, бесплатный для использования</p>
        </div>
      </div>
    </div>
  </div>

  <footer class="bg-[#29465b] text-white py-6 px-8 flex items-center justify-between">
    <div class="text-sm">
      <p>© 2025 A1Squad</p>
      <p>All Rights Reserved</p>
    </div>
    <a href="https://github.com/Alyukin/AISight" target="_blank" class="bg-white rounded-md p-2 hover:bg-gray-200 transition">
      <img src="/github-icon.svg" alt="GitHub" class="w-6 h-6" />
    </a>
  </footer>
</template>

