if (!window.dash_clientside) {
    window.dash_clientside = {};
}
window.dash_clientside.clientside = {
    make_draggable: function(id) {
        setTimeout(function() {
            var el = document.getElementById(id)
            //window.console.log(el)
            dragula([el])
        }, 1)
        return window.dash_clientside.no_update
    }
}



/*



var tagsboard = document.querySelector('.row');
console.log('cosa')
console.log(tagsboard)
console.log('fin cosa')
var cells = tagsboard.querySelectorAll('.cell');
var boxes = tagsboard.querySelectorAll('.box');
var dragCorners = tagsboard.querySelectorAll('.box .draggable');

var isDragged = false;
var draggedBox = null;
var dragStartX = 0;
var dragStartY = 0;
var dragX = 0;
var dragY = 0;
var cellToDrop = null;
var scrollTop = 0;


function getRotation (startHeight, targetHeight, perspective) {
  return targetHeight>=startHeight ? 0 :
  180 - (
    Math.PI*2 -
    (Math.PI/2 + Math.atan(targetHeight/perspective)) -
    Math.asin(Math.sin(Math.PI/2+Math.atan(targetHeight/perspective))*targetHeight/startHeight)
  ) * 180 / Math.PI;
}


function adaptHeights() {
  // TODO doladit pro adaptaci z větší na menší
  
  cells.forEach(cell => {
    var startHeight = getComputedStyle(cell).getPropertyValue('height');
    cell.style.removeProperty('height');
    var endHeight = getComputedStyle(cell).getPropertyValue('height');

    if (cell.animate) {
      cell.animate([
        {height: startHeight},
        {height: endHeight}
      ], 300);
    }
  });
}


function bindDragStart (event) {
  var box = event.target.parentNode;
  box.classList.add('dragged');
  
  isDragged = true;
  draggedBox = box;
  dragStartX = event.type === 'mousedown' ? event.pageX : event.touches[0].pageX;
  dragStartY = event.type === 'mousedown' ? event.pageY : event.touches[0].pageY;
  dragX = 0;
  dragY = 0;
 
  cells.forEach(cell => cell.style.height = `${cell.offsetHeight}px`);
  boxes.forEach(box => box.classList.remove('shift-preview-end'));
}


function bindDragMove (event) {
  if (isDragged) {
    if (event.type === 'mousemove') {
      event.preventDefault();
    } else {
      // TODO prevent scrolling when dragging
    }

    dragX = (event.type === 'mousemove' ? event.pageX : event.touches[0].pageX) - dragStartX;
    dragY = (event.type === 'mousemove' ? event.pageY : event.touches[0].pageY) - dragStartY;
    draggedBox.style.transform = `translate(${dragX}px, ${dragY}px)`;
    
    var boxCoords = draggedBox.getBoundingClientRect();
    var boxCenterX = boxCoords.left + boxCoords.width/2;
    var boxCenterY = boxCoords.top + scrollTop;
    
    cellsToDrop = [...cells]
      .map((cell, index) => {
        var cellCoords = cell.getBoundingClientRect();
        var cellCenterX = cellCoords.left + cellCoords.width/2;
        var cellCenterY = cellCoords.top + scrollTop;
        
        return {
          cell: cell,
          distanceX: Math.abs(cellCenterX-boxCenterX),
          distanceY: Math.abs(cellCenterY-boxCenterY),
          distance: Math.sqrt(Math.pow(cellCenterX-boxCenterX, 2) + Math.pow(cellCenterY-boxCenterY, 2))
        };
      })
      .sort((a, b) => {return a.distance - b.distance})
      .filter(cell => {return cell.distanceX < cell.cell.offsetWidth/2*.9 && cell.distanceY < cell.cell.offsetHeight*.9});
    
    cells.forEach(cell => cell.classList.remove('drop-here'));
    if (cellsToDrop.length > 0) {
      if (cellToDrop && cellToDrop !== cellsToDrop[0].cell) {
        endSwapPreview();
      }

      cellToDrop = cellsToDrop[0].cell;
      previewSwap(cellToDrop, draggedBox.parentNode);
      cellToDrop.classList.add('drop-here');
    } else {
      cellToDrop = null;
      endSwapPreview();
    }
  }
}


function previewSwap (from, to) {
  var box = from.querySelector('.box');
  
  if (!box || box === draggedBox) {
    return;
  }
  
  box.classList.add('shift-preview');
  var startCoords = from.getBoundingClientRect();
  var startStyle = getComputedStyle(from);
  var startInnerHeight = startCoords.height - parseInt(startStyle.paddingTop) - parseInt(startStyle.paddingBottom);

  var targetCoords = to.getBoundingClientRect();
  var targetStyle = getComputedStyle(to);
  var targetInnerHeight = targetCoords.height - parseInt(targetStyle.paddingTop) - parseInt(targetStyle.paddingBottom);

  var moveX = (targetCoords.left + parseInt(targetStyle.paddingLeft)) - (startCoords.left + parseInt(startStyle.paddingLeft));
  var moveY = (targetCoords.top + parseInt(targetStyle.paddingTop) + scrollTop) - (startCoords.top + parseInt(startStyle.paddingTop) + scrollTop);
  var perspective = 800;
  var rotateX = getRotation(startInnerHeight, targetInnerHeight, perspective);
  
  box.style.transform = `translate(${moveX}px, ${moveY}px) perspective(${perspective}px) rotateX(${rotateX}deg)`;
}


function endSwapPreview () {
  tagsboard.querySelectorAll('.shift-preview').forEach(box => {
    box.classList.remove('shift-preview');
    box.classList.add('shift-preview-end');
    box.style.removeProperty('transform');
  });
}


function dropToTarget () {
  if (draggedBox && cellToDrop) {
    var startHeight = draggedBox.offsetHeight;    
    cellToDrop.appendChild(draggedBox);
    draggedBox.style.removeProperty('transform');
    
    if (draggedBox.animate) {
      var lastX = dragStartX + dragX;
      var lastY = dragStartY + dragY;

      var targetCoords = draggedBox.getBoundingClientRect();
      var corner = draggedBox.querySelector('.draggable');

      var targetHeight = draggedBox.offsetHeight;
      var moveX = lastX - (targetCoords.left + corner.offsetLeft + corner.offsetWidth/2);
      var moveY = lastY - (targetCoords.top + scrollTop + corner.offsetTop + corner.offsetHeight/2);

      draggedBox.classList.add('dropped-to-target');

      var animation = draggedBox.animate([
        {transform: `translate(${moveX}px, ${moveY}px)`, height: `${startHeight}px`},
        {transform: `translate(0px, 0px)`, height: `${targetHeight}px`}
      ], {
        easing: 'cubic-bezier(0.520, 0.005, 0.585, 1.265)',
        duration: 150
      });
      animation.onfinish = () => {
        draggedBox.classList.remove('dropped-to-target');
        adaptHeights();
      };
    } else {
      adaptHeights();
    }
  }
}


function bindDragEnd (event) {
  if (isDragged) {
    if (cellToDrop) {
      var boxToSwitch = cellToDrop.querySelector('.box');
      if (boxToSwitch) {
        boxToSwitch.style.removeProperty('transform');
        draggedBox.parentNode.appendChild(boxToSwitch);
      }
      dropToTarget();
    } else {
      draggedBox.classList.add('dropped')
      draggedBox.style.removeProperty('transform');
    }
    
    isDragged = false;
    endSwapPreview();
    draggedBox.classList.remove('dragged');
    cells.forEach(cell => cell.classList.remove('drop-here'));
  }
}


function bindBoxTransitionEnd (event) {
  if (event.target.classList.contains('dropped') && /transform/.test(event.propertyName)) {
    event.target.classList.remove('dropped');
  }

  if (event.target.classList.contains('dropped-to-target') && /transform/.test(event.propertyName)) {
    event.target.classList.remove('dropped-to-target');
  }

  if (event.target.classList.contains('shift-preview-end') && /transform/.test(event.propertyName)) {
    event.target.classList.remove('shift-preview-end');
  }

  // cells.forEach(cell => cell.classList.remove('drop-here'));
}


function bindWindowScroll (event) {
  scrollTop = (window.pageYOffset !== undefined) ? window.pageYOffset : (document.documentElement || document.body.parentNode || document.body).scrollTop
}
bindWindowScroll();


dragCorners.forEach(corner => corner.addEventListener('mousedown', bindDragStart));
dragCorners.forEach(corner => corner.addEventListener('touchstart', bindDragStart));
document.addEventListener('mousemove', bindDragMove);
document.addEventListener('touchmove', bindDragMove);
document.addEventListener('mouseup', bindDragEnd);
document.addEventListener('touchend', bindDragEnd);

boxes.forEach(box => box.addEventListener('transitionend', bindBoxTransitionEnd));

window.addEventListener('scroll', bindWindowScroll);*/