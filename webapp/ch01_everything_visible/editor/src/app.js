/* 카드뉴스 편집기 (실습용 원본 코드)
 * 이 파일은 그대로 배포하지 않아요. webapp_ch01_build.py가 주석과 빈칸을 지우고
 * 이름을 짧게 바꾼 app.min.js를 만들어 public/ 폴더에 넣어요.
 */

// 템플릿 목록은 서버의 templates/index.json에서 읽어요.
const TEMPLATE_INDEX = "templates/index.json";
// AI 배경은 브라우저가 직접 만들지 않고 우리 서버에 부탁해요. 키는 서버에만 있어요.
const GENERATE_ENDPOINT = "/api/generate-image";

let currentTemplate = null;

async function loadTemplateList() {
  const response = await fetch(TEMPLATE_INDEX);
  const data = await response.json();
  const select = document.getElementById("template");
  for (const item of data.templates) {
    const option = document.createElement("option");
    option.value = item.file;
    option.textContent = item.name;
    select.appendChild(option);
  }
  select.addEventListener("change", function () { applyTemplate(select.value); });
  await applyTemplate(data.templates[0].file);
}

async function applyTemplate(templateFile) {
  const response = await fetch(templateFile);
  currentTemplate = await response.json();
  renderCard();
}

function renderCard() {
  const card = document.getElementById("card");
  const titleText = document.getElementById("title").value.slice(0, currentTemplate.title.maxChars);
  const bodyText = document.getElementById("body").value.slice(0, currentTemplate.body.maxChars);
  card.innerHTML = "";
  const heading = document.createElement("h2");
  heading.textContent = titleText;
  heading.style.color = currentTemplate.title.color;
  const paragraph = document.createElement("p");
  paragraph.textContent = bodyText;
  paragraph.style.color = currentTemplate.body.color;
  card.appendChild(heading);
  card.appendChild(paragraph);
}

// TODO: 사용량 제한은 아직 브라우저에서만 세고 있어요. 서버에서도 세도록 옮기기.
async function requestBackground() {
  const userPrompt = document.getElementById("prompt").value;
  const response = await fetch(GENERATE_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: userPrompt, template: currentTemplate.id })
  });
  const svgText = await response.text();
  const card = document.getElementById("card");
  card.style.backgroundImage = "url(\"data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgText) + "\")";
}

document.getElementById("title").addEventListener("input", renderCard);
document.getElementById("body").addEventListener("input", renderCard);
document.getElementById("make-bg").addEventListener("click", requestBackground);
loadTemplateList();
